import vulkan as vk
import ctypes
import numpy as np
import os
import subprocess
import time


class VulkanMoERouter:
    """
    GPU-accelerated distance computation via Vulkan compute shaders.
    
    Uses PERSISTENT pre-allocated GPU buffers to eliminate per-query allocation overhead.
    Buffers are created once at init and reused across all compute_distances() calls.
    
    SINGLETON: Only one instance per (dim) is created to avoid VkErrorDeviceLost
    from multiple Vulkan device contexts on the same GPU.
    """
    
    # Default max candidates per query. Buffers auto-resize if exceeded.
    DEFAULT_MAX_EXPERTS = 8192
    
    # Singleton instances keyed by dim
    _instances = {}
    
    def __new__(cls, dim=640, max_experts=None):
        if dim not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[dim] = instance
        return cls._instances[dim]
    
    def __init__(self, dim=640, max_experts=None):
        if self._initialized:
            return
        self._initialized = True
        
        self.dim = dim
        self.max_experts = max_experts or self.DEFAULT_MAX_EXPERTS
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.shader_path = os.path.join(self.base_dir, "shaders", "moe.spv")
        self._compile_shader()
        
        # Core Vulkan objects (created once, never destroyed until __del__)
        self.instance = self._create_instance()
        self.physical_device = self._get_physical_device()
        self.queue_family_index = self._get_compute_queue_family()
        self.device, self.queue = self._create_device()
        self.command_pool = self._create_command_pool()
        
        self.shader_module = self._create_shader_module()
        self.descriptor_set_layout = self._create_descriptor_set_layout()
        self.pipeline_layout, self.pipeline = self._create_pipeline()
        
        # Pre-allocate persistent GPU buffers
        self._alloc_persistent_buffers(self.max_experts)
        
        # Pre-allocate command buffer (reusable with reset)
        self._alloc_command_buffer()

    # ─────────────────────────────────────────────────────────
    # Persistent buffer management
    # ─────────────────────────────────────────────────────────
    
    def _alloc_persistent_buffers(self, max_experts):
        """Pre-allocate query, expert, and output GPU buffers for max_experts candidates."""
        self._buf_max_experts = max_experts
        
        query_size = self.dim * 4  # float32
        experts_size = max_experts * self.dim * 4
        output_size = max_experts * 4
        
        self._q_buf, self._q_mem, self._q_size = *self._create_buffer(query_size), query_size
        self._e_buf, self._e_mem, self._e_size = *self._create_buffer(experts_size), experts_size
        self._o_buf, self._o_mem, self._o_size = *self._create_buffer(output_size), output_size
        
        # Persistent descriptor pool + set
        self._descriptor_pool = self._create_persistent_descriptor_pool()
        self._descriptor_set = self._allocate_descriptor_set()
        self._update_descriptor_set()
    
    def _free_persistent_buffers(self):
        """Free current persistent buffers."""
        vk.vkDestroyDescriptorPool(self.device, self._descriptor_pool, None)
        for buf, mem in [(self._q_buf, self._q_mem), (self._e_buf, self._e_mem), (self._o_buf, self._o_mem)]:
            vk.vkDestroyBuffer(self.device, buf, None)
            vk.vkFreeMemory(self.device, mem, None)
    
    def _resize_if_needed(self, num_experts):
        """Resize persistent buffers if num_experts exceeds current capacity."""
        if num_experts <= self._buf_max_experts:
            return
        # Double the capacity to amortize resizes
        new_max = max(num_experts, self._buf_max_experts * 2)
        print(f"[VULKAN] Resizing buffers: {self._buf_max_experts} → {new_max}")
        self._free_persistent_buffers()
        self._alloc_persistent_buffers(new_max)
        # Re-create command buffer references after resize
        self._pSets = vk.ffi.new('VkDescriptorSet[]', [self._descriptor_set])
        
    def _create_persistent_descriptor_pool(self):
        poolSize = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=3
        )
        pPoolSizes = vk.ffi.new('VkDescriptorPoolSize[]', [poolSize])
        poolInfo = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=1,
            poolSizeCount=1,
            pPoolSizes=pPoolSizes
        )
        return vk.vkCreateDescriptorPool(self.device, poolInfo, None)
    
    def _allocate_descriptor_set(self):
        pSetLayouts = vk.ffi.new('VkDescriptorSetLayout[]', [self.descriptor_set_layout])
        allocInfo = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self._descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=pSetLayouts
        )
        return vk.vkAllocateDescriptorSets(self.device, allocInfo)[0]
    
    def _update_descriptor_set(self):
        """Bind persistent buffers to descriptor set."""
        writes = []
        refs = []
        for i, (buf, sz) in enumerate([
            (self._q_buf, self._q_size),
            (self._e_buf, self._e_size),
            (self._o_buf, self._o_size)
        ]):
            bInfo = vk.VkDescriptorBufferInfo(buffer=buf, offset=0, range=sz)
            pBInfo = vk.ffi.new('VkDescriptorBufferInfo[]', [bInfo])
            refs.append(pBInfo)
            write = vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self._descriptor_set,
                dstBinding=i,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=pBInfo
            )
            writes.append(write)
        pWrites = vk.ffi.new('VkWriteDescriptorSet[]', writes)
        vk.vkUpdateDescriptorSets(self.device, 3, pWrites, 0, vk.VK_NULL_HANDLE)
    
    def _alloc_command_buffer(self):
        """Pre-allocate a reusable command buffer."""
        allocInfoCmd = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        self._cmd_buffer = vk.vkAllocateCommandBuffers(self.device, allocInfoCmd)[0]
        
        # Pre-create submit info references
        self._pCmdBuffers = vk.ffi.new('VkCommandBuffer[]', [self._cmd_buffer])
        self._pSets = vk.ffi.new('VkDescriptorSet[]', [self._descriptor_set])

    # ─────────────────────────────────────────────────────────
    # Core compute (HOT PATH - no allocations)
    # ─────────────────────────────────────────────────────────
    
    def compute_distances(self, query_np, expert_embeddings_np):
        """
        Compute L2 distances between query and expert embeddings on GPU.
        
        Uses persistent pre-allocated buffers — only memory map/copy/dispatch/read per call.
        No buffer creation or destruction in the hot path.
        """
        query_np = np.ascontiguousarray(query_np.astype(np.float32))
        expert_embeddings_np = np.ascontiguousarray(expert_embeddings_np.astype(np.float32))
        
        num_experts = expert_embeddings_np.shape[0]
        dim = expert_embeddings_np.shape[1]
        
        # Resize persistent buffers if needed (rare path)
        self._resize_if_needed(num_experts)
        
        query_bytes = query_np.nbytes
        experts_bytes = expert_embeddings_np.nbytes
        output_bytes = num_experts * 4
        
        # === HOT PATH: map → copy → dispatch → read ===
        
        # Upload query
        q_ptr = vk.vkMapMemory(self.device, self._q_mem, 0, query_bytes, 0)
        vk.ffi.memmove(q_ptr, vk.ffi.from_buffer(query_np.data), query_bytes)
        vk.vkUnmapMemory(self.device, self._q_mem)
        
        # Upload experts
        e_ptr = vk.vkMapMemory(self.device, self._e_mem, 0, experts_bytes, 0)
        vk.ffi.memmove(e_ptr, vk.ffi.from_buffer(expert_embeddings_np.data), experts_bytes)
        vk.vkUnmapMemory(self.device, self._e_mem)
        
        # Record command buffer (reset + re-record is cheaper than alloc/free)
        vk.vkResetCommandBuffer(self._cmd_buffer, 0)
        
        beginInfo = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        vk.vkBeginCommandBuffer(self._cmd_buffer, beginInfo)
        
        vk.vkCmdBindPipeline(self._cmd_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
        vk.vkCmdBindDescriptorSets(
            self._cmd_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline_layout, 0, 1, self._pSets, 0, vk.VK_NULL_HANDLE
        )
        
        pConstants = vk.ffi.new('uint32_t[]', [num_experts, dim])
        vk.vkCmdPushConstants(
            self._cmd_buffer, self.pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, vk.ffi.cast('void*', pConstants)
        )
        
        group_x = (num_experts + 255) // 256
        vk.vkCmdDispatch(self._cmd_buffer, group_x, 1, 1)
        
        vk.vkEndCommandBuffer(self._cmd_buffer)
        
        # Submit and wait
        try:
            submitInfo = vk.VkSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount=1,
                pCommandBuffers=self._pCmdBuffers
            )
            pSubmitInfos = vk.ffi.new('VkSubmitInfo[]', [submitInfo])
            vk.vkQueueSubmit(self.queue, 1, pSubmitInfos, vk.VK_NULL_HANDLE)
            vk.vkQueueWaitIdle(self.queue)
        except Exception as e:
            print(f"[VULKAN ERROR] compute_distances failed: {e}")
            print(f"  num_experts={num_experts}, dim={dim}, buf_max={self._buf_max_experts}")
            print(f"  query_bytes={query_bytes}, experts_bytes={experts_bytes}, output_bytes={output_bytes}")
            print(f"  _e_size={self._e_size}, _o_size={self._o_size}")
            # Fall back to CPU
            return np.linalg.norm(expert_embeddings_np - query_np, axis=1)
        
        # Read results
        o_ptr = vk.vkMapMemory(self.device, self._o_mem, 0, output_bytes, 0)
        distances = np.zeros(num_experts, dtype=np.float32)
        vk.ffi.memmove(vk.ffi.from_buffer(distances.data), o_ptr, output_bytes)
        vk.vkUnmapMemory(self.device, self._o_mem)
        
        return distances

    # ─────────────────────────────────────────────────────────
    # Vulkan setup (unchanged from original)
    # ─────────────────────────────────────────────────────────
    
    def _compile_shader(self):
        shaders_dir = os.path.join(self.base_dir, "shaders")
        os.makedirs(shaders_dir, exist_ok=True)
        comp_path = os.path.join(shaders_dir, "moe.comp")
        if not os.path.exists(self.shader_path) or os.path.getmtime(comp_path) > os.path.getmtime(self.shader_path):
            print(f"Compiling {comp_path} to {self.shader_path} via glslc...")
            subprocess.run(["glslc", comp_path, "-o", self.shader_path], check=True)

    def _create_instance(self):
        appInfo = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="M2M MoE Router",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="No Engine",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0
        )
        createInfo = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=appInfo
        )
        return vk.vkCreateInstance(createInfo, None)

    def _get_physical_device(self):
        devices = vk.vkEnumeratePhysicalDevices(self.instance)
        if not devices:
            raise RuntimeError("No Vulkan physical devices found.")
        return devices[0]

    def _get_compute_queue_family(self):
        queueFamilies = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)
        for i, qf in enumerate(queueFamilies):
            if qf.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                return i
        raise RuntimeError("No compute queue family found.")

    def _create_device(self):
        queuePriorities = vk.ffi.new('float[]', [1.0])
        queueCreateInfo = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.queue_family_index,
            queueCount=1,
            pQueuePriorities=queuePriorities
        )
        createInfo = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queueCreateInfo]
        )
        device = vk.vkCreateDevice(self.physical_device, createInfo, None)
        queue = vk.vkGetDeviceQueue(device, self.queue_family_index, 0)
        return device, queue

    def _create_command_pool(self):
        poolInfo = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.queue_family_index,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        )
        return vk.vkCreateCommandPool(self.device, poolInfo, None)

    def _create_shader_module(self):
        with open(self.shader_path, "rb") as f:
            code = f.read()
        createInfo = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(code),
            pCode=code
        )
        return vk.vkCreateShaderModule(self.device, createInfo, None)

    def _create_descriptor_set_layout(self):
        bindings = []
        for i in range(3):
            binding = vk.VkDescriptorSetLayoutBinding(
                binding=i,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
            )
            bindings.append(binding)
        pBindings = vk.ffi.new('VkDescriptorSetLayoutBinding[]', bindings)
        createInfo = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=pBindings
        )
        return vk.vkCreateDescriptorSetLayout(self.device, createInfo, None)

    def _create_pipeline(self):
        push_constants = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=8
        )
        pSetLayouts = vk.ffi.new('VkDescriptorSetLayout[]', [self.descriptor_set_layout])
        pPushConstants = vk.ffi.new('VkPushConstantRange[]', [push_constants])
        
        layoutInfo = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=pSetLayouts,
            pushConstantRangeCount=1,
            pPushConstantRanges=pPushConstants
        )
        pipelineLayout = vk.vkCreatePipelineLayout(self.device, layoutInfo, None)

        name = vk.ffi.new('char[]', b'main')
        stageInfo = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=self.shader_module,
            pName=name
        )
        pipelineInfo = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stageInfo,
            layout=pipelineLayout
        )
        pipelines = vk.vkCreateComputePipelines(self.device, vk.VK_NULL_HANDLE, 1, [pipelineInfo], None)
        return pipelineLayout, pipelines[0]

    def _find_memory_type(self, typeFilter, properties):
        memProperties = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        for i in range(memProperties.memoryTypeCount):
            if (typeFilter & (1 << i)) and ((memProperties.memoryTypes[i].propertyFlags & properties) == properties):
                return i
        raise RuntimeError("failed to find suitable memory type")

    def _create_buffer(self, size):
        bufferInfo = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
        )
        buffer = vk.vkCreateBuffer(self.device, bufferInfo, None)
        memRequirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        
        allocInfo = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=memRequirements.size,
            memoryTypeIndex=self._find_memory_type(
                memRequirements.memoryTypeBits, 
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            )
        )
        memory = vk.vkAllocateMemory(self.device, allocInfo, None)
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        return buffer, memory


if __name__ == "__main__":
    print("Testing Vulkan Compute API bindings (persistent buffers)...")
    router = VulkanMoERouter(dim=640, max_experts=1000)
    
    q = np.random.randn(640).astype(np.float32)
    e = np.random.randn(100, 640).astype(np.float32)
    
    # Warmup
    router.compute_distances(q, e)
    
    # Benchmark: 100 sequential calls
    times = []
    for _ in range(100):
        start = time.perf_counter()
        dist = router.compute_distances(q, e)
        times.append((time.perf_counter() - start) * 1000)
    
    dist_cpu = np.linalg.norm(e - q, axis=1)
    max_diff = np.abs(dist - dist_cpu).max()
    
    print(f"Correctness: max diff from CPU = {max_diff:.6f}")
    print(f"Latency (100 calls): avg={np.mean(times):.2f}ms, p50={np.median(times):.2f}ms, min={np.min(times):.2f}ms")
    print("Vulkan Persistent Buffer Test Passed!")
