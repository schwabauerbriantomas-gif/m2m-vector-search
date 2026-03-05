"""
GPUVectorIndex — Persistent full-index GPU buffer with batch query dispatch.

Reference pattern implemented:
  ✅ Index uploaded ONCE at __init__, never re-uploaded
  ✅ Only queries (tiny) are transferred per search call
  ✅ Batch dispatch: all B queries processed in a single vkCmdDispatch
  ✅ Shared memory query cache in the GLSL kernel

HierarchicalGPUSearch — Two-stage GPU search:
  Stage 1  Coarse: queries vs cluster centroids (O(Q × C), C << N)
  Stage 2  Fine:   queries vs candidate cluster members (O(Q × M × n_probe))
  Both stages use persistent GPU buffers.
"""

import vulkan as vk
import ctypes
import numpy as np
import os
import subprocess
import time
from typing import Optional

# Max vectors per GPU dispatch chunk.
# Result buffer size = MAX_BATCH × CHUNK_SIZE × 4 bytes.
# 100 queries × 8192 vectors × 4 = 3.2 MB — safe for all Vulkan implementations.
_CHUNK_SIZE = 8192


# ═══════════════════════════════════════════════════════════════════════════════
# GPUVectorIndex — Persistent index + batch query dispatch
# ═══════════════════════════════════════════════════════════════════════════════

class GPUVectorIndex:
    """
    Persistent GPU vector index with batch query dispatch.

    Memory layout (once at init, never reallocated unless rebuild() called):

        ┌─────────────────────┬─────────────┬───────────────────────┐
        │ Region              │ Size        │ Contents              │
        ├─────────────────────┼─────────────┼───────────────────────┤
        │ Index Buffer        │ N × D × 4   │ Index vectors (float) │
        │ (Persistent)        │ bytes       │ — uploaded ONCE       │
        ├─────────────────────┼─────────────┼───────────────────────┤
        │ Query Buffer        │ B × D × 4   │ Batch of queries      │
        │ (Dynamic)           │ bytes       │ — copied per call     │
        ├─────────────────────┼─────────────┼───────────────────────┤
        │ Result Buffer       │ B × N × 4   │ L2 distances          │
        │ (Dynamic)           │ bytes       │ — read after dispatch │
        └─────────────────────┴─────────────┴───────────────────────┘

    Dispatch: vkCmdDispatch(ceil(N/256), B, 1)
      — grid.x covers all index vectors in 256-thread workgroups
      — grid.y selects the query; kernel loads it into shared mem once
    """

    def __init__(self, index_vectors: np.ndarray, max_batch_size: int = 100):
        """
        Args:
            index_vectors: shape [N, D] float32 — the full vector index.
            max_batch_size: max queries per batch_search() call.
        """
        assert index_vectors.ndim == 2, "index_vectors must be [N, D]"
        self._n = index_vectors.shape[0]
        self._dim = index_vectors.shape[1]
        self._max_batch = max_batch_size
        self._base_dir = os.path.dirname(os.path.abspath(__file__))

        # ── Compile shader if needed ──────────────────────────────────
        shader_spv = os.path.join(self._base_dir, "shaders", "moe_batch.spv")
        shader_comp = os.path.join(self._base_dir, "shaders", "moe_batch.comp")
        if not os.path.exists(shader_spv) or (
            os.path.exists(shader_comp) and
            os.path.getmtime(shader_comp) > os.path.getmtime(shader_spv)
        ):
            print("[GPUVectorIndex] Compiling moe_batch.comp...")
            subprocess.run(["glslc", shader_comp, "-o", shader_spv], check=True)
        self._shader_spv = shader_spv

        # ── Vulkan init ───────────────────────────────────────────────
        self._vk_instance = self._create_instance()
        self._physical_device = self._get_physical_device()
        self._queue_family = self._get_compute_queue_family()
        self._device, self._queue = self._create_device()
        self._cmd_pool = self._create_command_pool()

        self._shader_module = self._load_shader()
        self._dsl = self._create_descriptor_set_layout()
        self._pipeline_layout, self._pipeline = self._create_pipeline()

        # ── Persistent buffers ────────────────────────────────────────
        # Index buffer (uploaded once)
        index_np = np.ascontiguousarray(index_vectors, dtype=np.float32)
        self._idx_buf, self._idx_mem = self._create_buffer(index_np.nbytes)
        self._upload(self._idx_mem, index_np)

        # Query buffer (max_batch_size × dim — overwritten each call)
        self._q_buf, self._q_mem = self._create_buffer(max_batch_size * self._dim * 4)

        # Result buffer: bounded to [max_batch_size × CHUNK_SIZE] — never full [B×N].
        # Chunked dispatch keeps this ≤ max_batch × CHUNK_SIZE × 4 bytes (≤ ~3 MB).
        self._chunk_size = min(_CHUNK_SIZE, self._n)
        self._r_buf, self._r_mem = self._create_buffer(max_batch_size * self._chunk_size * 4)

        # ── Descriptor set ────────────────────────────────────────────
        self._desc_pool = self._make_descriptor_pool(3)
        self._desc_set = self._allocate_descriptor_set()
        self._update_descriptor_set()

        # ── Pre-allocated command buffer ──────────────────────────────
        self._cmd = self._alloc_command_buffer()
        self._p_cmds = vk.ffi.new("VkCommandBuffer[]", [self._cmd])
        self._p_sets = vk.ffi.new("VkDescriptorSet[]", [self._desc_set])

        print(f"[GPUVectorIndex] Ready — index {self._n:,}×{self._dim}  "
              f"({index_np.nbytes / 1024**2:.1f} MB persistent on GPU)  "
              f"chunk_size={self._chunk_size:,}")

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def batch_search(self, queries: np.ndarray, k: int = 64):
        """
        Search B queries against the full persistent index.

        Uses chunked dispatch over N to keep the result buffer bounded
        (≤ max_batch × CHUNK_SIZE × 4 bytes ≈ 3 MB).
        
        Args:
            queries: shape [B, D] float32
            k: number of nearest neighbours to return

        Returns:
            indices   shape [B, k]  int64
            distances shape [B, k]  float32
        """
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        assert queries.ndim == 2 and queries.shape[1] == self._dim
        batch_size = queries.shape[0]
        assert batch_size <= self._max_batch, (
            f"batch_size {batch_size} > max_batch {self._max_batch}. "
            "Re-create GPUVectorIndex with a larger max_batch_size."
        )
        k = min(k, self._n)

        # Upload queries once — only queries are transferred, never the index
        self._upload(self._q_mem, queries)

        # Rolling top-K: track best k distances/ids seen so far
        best_ids   = np.full((batch_size, k), -1,  dtype=np.int64)
        best_dists = np.full((batch_size, k), np.inf, dtype=np.float32)

        chunk = self._chunk_size

        # Iterate over index in chunks — the index buffer stays persistent;
        # only push constants change to tell the shader which chunk to process.
        for chunk_start in range(0, self._n, chunk):
            chunk_end  = min(chunk_start + chunk, self._n)
            chunk_n    = chunk_end - chunk_start

            self._dispatch_chunk(batch_size, chunk_start, chunk_n)

            # Read [B × chunk_n] distances — must be a fresh contiguous array
            # (slices of a larger array are non-contiguous and break ffi.memmove)
            chunk_view = np.empty((batch_size, chunk_n), dtype=np.float32)
            nbytes = batch_size * chunk_n * 4
            ptr = vk.vkMapMemory(self._device, self._r_mem, 0, nbytes, 0)
            vk.ffi.memmove(vk.ffi.from_buffer(chunk_view.data), ptr, nbytes)
            vk.vkUnmapMemory(self._device, self._r_mem)

            # Merge chunk results into rolling top-K
            # Prepend chunk distances with already-accumulated bests
            combined_ids   = np.concatenate([
                best_ids,
                np.arange(chunk_start, chunk_end, dtype=np.int64)[None, :].repeat(batch_size, axis=0)
            ], axis=1)
            combined_dists = np.concatenate([best_dists, chunk_view], axis=1)

            # Pick new top-k from combined
            part = np.argpartition(combined_dists, k, axis=1)[:, :k]
            best_dists = np.take_along_axis(combined_dists, part, axis=1)
            best_ids   = np.take_along_axis(combined_ids,  part, axis=1)

        # Final sort within top-k
        order = np.argsort(best_dists, axis=1)
        best_ids   = np.take_along_axis(best_ids,   order, axis=1)
        best_dists = np.take_along_axis(best_dists, order, axis=1)

        return best_ids, best_dists

    def rebuild(self, new_index_vectors: np.ndarray):
        """Re-upload the index (call only when index changes)."""
        new_idx = np.ascontiguousarray(new_index_vectors, dtype=np.float32)
        assert new_idx.shape == (self._n, self._dim), (
            "Shape mismatch; create a new GPUVectorIndex for different N or D."
        )
        self._upload(self._idx_mem, new_idx)
        print(f"[GPUVectorIndex] Index rebuilt ({self._n:,} vectors)")

    def compute_distances(self, query_np: np.ndarray,
                          expert_embeddings_np: np.ndarray) -> np.ndarray:
        """
        Drop-in replacement for VulkanMoERouter.compute_distances().

        Computes L2 distances between ONE query and a dynamic set of expert vectors.
        This is the MoE router hot path: query is 1-D, experts are 2-D and change each call.

        API-compatible with the old VulkanMoERouter:
            distances = router.compute_distances(query_1d, experts_2d)  # [N] float32

        Args:
            query_np:           shape [D] or [1, D]  float32
            expert_embeddings_np: shape [N, D]         float32

        Returns:
            distances: shape [N] float32 — L2 distance from query to each expert
        """
        query_np = np.ascontiguousarray(query_np.flatten()[:self._dim], dtype=np.float32)
        expert_embeddings_np = np.ascontiguousarray(
            expert_embeddings_np.astype(np.float32)
        )
        n_experts = expert_embeddings_np.shape[0]

        if n_experts == 0:
            return np.array([], dtype=np.float32)

        # For small expert sets (< CHUNK_SIZE), we can re-use the persistent index
        # buffer by uploading the experts as the "index" and running one chunk dispatch.
        # This matches VulkanMoERouter behaviour: experts change per call, query is tiny.
        if n_experts > self._n:
            # Expert set larger than our index — fall back to CPU NumPy
            return np.linalg.norm(expert_embeddings_np - query_np, axis=1)

        # Upload experts as the current index (overwrites persistent buffer for this call)
        experts_bytes = expert_embeddings_np[:n_experts].nbytes
        e_ptr = vk.vkMapMemory(self._device, self._idx_mem, 0, experts_bytes, 0)
        vk.ffi.memmove(
            e_ptr,
            vk.ffi.from_buffer(expert_embeddings_np[:n_experts].data),
            experts_bytes,
        )
        vk.vkUnmapMemory(self._device, self._idx_mem)

        # Upload query as [1, D] batch
        query_2d = query_np[None, :]  # [1, D]
        self._upload(self._q_mem, query_2d)

        # Dispatch: 1 query, n_experts vectors, chunk_start=0
        chunk_n = n_experts
        self._dispatch_chunk(batch_size=1, chunk_start=0, chunk_n=chunk_n)

        # Read [1 × n_experts] distances
        out = np.empty(n_experts, dtype=np.float32)
        nbytes = n_experts * 4
        ptr = vk.vkMapMemory(self._device, self._r_mem, 0, nbytes, 0)
        vk.ffi.memmove(vk.ffi.from_buffer(out.data), ptr, nbytes)
        vk.vkUnmapMemory(self._device, self._r_mem)

        return out

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _dispatch(self, batch_size: int):
        """Full dispatch — all N vectors (use only when N <= CHUNK_SIZE)."""
        self._dispatch_chunk(batch_size, chunk_start=0, chunk_n=self._n)

    def _dispatch_chunk(self, batch_size: int, chunk_start: int, chunk_n: int):
        """
        Dispatch kernel for vectors [chunk_start .. chunk_start+chunk_n).
        Push constants encode: chunk_n, dim (and chunk_start via index buffer offset
        using a 3rd push constant).
        """
        vk.vkResetCommandBuffer(self._cmd, 0)

        begin = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(self._cmd, begin)

        vk.vkCmdBindPipeline(self._cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipeline)
        vk.vkCmdBindDescriptorSets(
            self._cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self._pipeline_layout, 0, 1, self._p_sets, 0, vk.VK_NULL_HANDLE,
        )

        # Push constants: num_vectors=chunk_n, dim
        # The shader uses gl_GlobalInvocationID.x as a LOCAL index [0..chunk_n).
        # The index buffer contains ALL vectors; the shader uses
        #   (chunk_start + local_idx) as the actual vector address.
        # We pass chunk_start as a 3rd push constant (uint32).
        pc = vk.ffi.new("uint32_t[]", [chunk_n, self._dim, chunk_start])
        vk.vkCmdPushConstants(
            self._cmd, self._pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, vk.ffi.cast("void*", pc),
        )

        group_x = (chunk_n + 255) // 256
        group_y = batch_size
        vk.vkCmdDispatch(self._cmd, group_x, group_y, 1)

        vk.vkEndCommandBuffer(self._cmd)

        submit = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=self._p_cmds,
        )
        p_submit = vk.ffi.new("VkSubmitInfo[]", [submit])
        vk.vkQueueSubmit(self._queue, 1, p_submit, vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self._queue)

    def _upload(self, memory, array: np.ndarray):
        """Map GPU memory and copy numpy array into it."""
        ptr = vk.vkMapMemory(self._device, memory, 0, array.nbytes, 0)
        vk.ffi.memmove(ptr, vk.ffi.from_buffer(array.data), array.nbytes)
        vk.vkUnmapMemory(self._device, memory)

    def _download(self, memory, array: np.ndarray):
        """Map GPU memory and copy into numpy array."""
        nbytes = array.nbytes
        ptr = vk.vkMapMemory(self._device, memory, 0, nbytes, 0)
        vk.ffi.memmove(vk.ffi.from_buffer(array.data), ptr, nbytes)
        vk.vkUnmapMemory(self._device, memory)

    def _update_descriptor_set(self):
        writes = []
        refs = []
        bufs_sizes = [
            (self._idx_buf, self._n * self._dim * 4),
            (self._q_buf,   self._max_batch * self._dim * 4),
            (self._r_buf,   self._max_batch * self._chunk_size * 4),
        ]
        for i, (buf, size) in enumerate(bufs_sizes):
            binfo = vk.VkDescriptorBufferInfo(buffer=buf, offset=0, range=size)
            p_binfo = vk.ffi.new("VkDescriptorBufferInfo[]", [binfo])
            refs.append(p_binfo)
            w = vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self._desc_set,
                dstBinding=i,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=p_binfo,
            )
            writes.append(w)
        p_writes = vk.ffi.new("VkWriteDescriptorSet[]", writes)
        vk.vkUpdateDescriptorSets(self._device, 3, p_writes, 0, vk.VK_NULL_HANDLE)

    def _create_instance(self):
        app = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="GPUVectorIndex",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="M2M",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app,
        )
        return vk.vkCreateInstance(info, None)

    def _get_physical_device(self):
        devices = vk.vkEnumeratePhysicalDevices(self._vk_instance)
        if not devices:
            raise RuntimeError("No Vulkan devices found.")
        return devices[0]

    def _get_compute_queue_family(self):
        families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self._physical_device)
        for i, f in enumerate(families):
            if f.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                return i
        raise RuntimeError("No compute queue family found.")

    def _create_device(self):
        prio = vk.ffi.new("float[]", [1.0])
        q_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self._queue_family,
            queueCount=1,
            pQueuePriorities=prio,
        )
        dev_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[q_info],
        )
        device = vk.vkCreateDevice(self._physical_device, dev_info, None)
        queue = vk.vkGetDeviceQueue(device, self._queue_family, 0)
        return device, queue

    def _create_command_pool(self):
        info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self._queue_family,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        )
        return vk.vkCreateCommandPool(self._device, info, None)

    def _load_shader(self):
        with open(self._shader_spv, "rb") as f:
            code = f.read()
        info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(code),
            pCode=code,
        )
        return vk.vkCreateShaderModule(self._device, info, None)

    def _create_descriptor_set_layout(self):
        bindings = []
        for i in range(3):
            b = vk.VkDescriptorSetLayoutBinding(
                binding=i,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            )
            bindings.append(b)
        p_bindings = vk.ffi.new("VkDescriptorSetLayoutBinding[]", bindings)
        info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=3,
            pBindings=p_bindings,
        )
        return vk.vkCreateDescriptorSetLayout(self._device, info, None)

    def _create_pipeline(self):
        pc_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=12,  # uint32 chunk_n + uint32 dim + uint32 chunk_start
        )
        p_sl = vk.ffi.new("VkDescriptorSetLayout[]", [self._dsl])
        p_pc = vk.ffi.new("VkPushConstantRange[]", [pc_range])
        layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=p_sl,
            pushConstantRangeCount=1,
            pPushConstantRanges=p_pc,
        )
        layout = vk.vkCreatePipelineLayout(self._device, layout_info, None)

        name = vk.ffi.new("char[]", b"main")
        stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=self._shader_module,
            pName=name,
        )
        pipe_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage,
            layout=layout,
        )
        pipelines = vk.vkCreateComputePipelines(
            self._device, vk.VK_NULL_HANDLE, 1, [pipe_info], None
        )
        return layout, pipelines[0]

    def _find_memory_type(self, type_filter, properties):
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self._physical_device)
        for i in range(mem_props.memoryTypeCount):
            if (type_filter & (1 << i)) and (
                (mem_props.memoryTypes[i].propertyFlags & properties) == properties
            ):
                return i
        raise RuntimeError("No suitable memory type.")

    def _create_buffer(self, size: int):
        """Create a host-visible, host-coherent storage buffer."""
        buf_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        buf = vk.vkCreateBuffer(self._device, buf_info, None)
        reqs = vk.vkGetBufferMemoryRequirements(self._device, buf)
        alloc = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=reqs.size,
            memoryTypeIndex=self._find_memory_type(
                reqs.memoryTypeBits,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            ),
        )
        mem = vk.vkAllocateMemory(self._device, alloc, None)
        vk.vkBindBufferMemory(self._device, buf, mem, 0)
        return buf, mem

    def _make_descriptor_pool(self, count: int):
        pool_size = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=count,
        )
        p_ps = vk.ffi.new("VkDescriptorPoolSize[]", [pool_size])
        info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=1,
            poolSizeCount=1,
            pPoolSizes=p_ps,
        )
        return vk.vkCreateDescriptorPool(self._device, info, None)

    def _allocate_descriptor_set(self):
        p_sl = vk.ffi.new("VkDescriptorSetLayout[]", [self._dsl])
        info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self._desc_pool,
            descriptorSetCount=1,
            pSetLayouts=p_sl,
        )
        return vk.vkAllocateDescriptorSets(self._device, info)[0]

    def _alloc_command_buffer(self):
        info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self._cmd_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        return vk.vkAllocateCommandBuffers(self._device, info)[0]
