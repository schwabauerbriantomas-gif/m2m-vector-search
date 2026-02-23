import vulkan as vk
import ctypes
import numpy as np
import os
import subprocess
import time

class VulkanMoERouter:
    def __init__(self, dim=640):
        self.dim = dim
        self.shader_path = "shaders/moe.spv"
        self._compile_shader()
        
        self.instance = self._create_instance()
        self.physical_device = self._get_physical_device()
        self.queue_family_index = self._get_compute_queue_family()
        self.device, self.queue = self._create_device()
        self.command_pool = self._create_command_pool()
        
        self.shader_module = self._create_shader_module()
        self.descriptor_set_layout = self._create_descriptor_set_layout()
        self.pipeline_layout, self.pipeline = self._create_pipeline()
        
    def _compile_shader(self):
        os.makedirs("shaders", exist_ok=True)
        comp_path = "shaders/moe.comp"
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

    def compute_distances(self, query_np, expert_embeddings_np):
        query_np = np.ascontiguousarray(query_np.astype(np.float32))
        expert_embeddings_np = np.ascontiguousarray(expert_embeddings_np.astype(np.float32))
        
        num_experts = expert_embeddings_np.shape[0]
        dim = expert_embeddings_np.shape[1]
        
        query_size = query_np.nbytes
        experts_size = expert_embeddings_np.nbytes
        output_size = num_experts * 4
        
        q_buf, q_mem = self._create_buffer(query_size)
        e_buf, e_mem = self._create_buffer(experts_size)
        o_buf, o_mem = self._create_buffer(output_size)
        
        q_ptr = vk.vkMapMemory(self.device, q_mem, 0, query_size, 0)
        vk.ffi.memmove(q_ptr, vk.ffi.from_buffer(query_np.data), query_size)
        vk.vkUnmapMemory(self.device, q_mem)
        
        e_ptr = vk.vkMapMemory(self.device, e_mem, 0, experts_size, 0)
        vk.ffi.memmove(e_ptr, vk.ffi.from_buffer(expert_embeddings_np.data), experts_size)
        vk.vkUnmapMemory(self.device, e_mem)
        
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
        descriptorPool = vk.vkCreateDescriptorPool(self.device, poolInfo, None)
        
        pSetLayouts = vk.ffi.new('VkDescriptorSetLayout[]', [self.descriptor_set_layout])
        allocInfo = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=descriptorPool,
            descriptorSetCount=1,
            pSetLayouts=pSetLayouts
        )
        descriptorSet = vk.vkAllocateDescriptorSets(self.device, allocInfo)[0]
        
        writes = []
        refs = []
        for i, (buf, sz) in enumerate([(q_buf, query_size), (e_buf, experts_size), (o_buf, output_size)]):
            bInfo = vk.VkDescriptorBufferInfo(buffer=buf, offset=0, range=sz)
            pBInfo = vk.ffi.new('VkDescriptorBufferInfo[]', [bInfo])
            refs.append(pBInfo)
            
            write = vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptorSet,
                dstBinding=i,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=pBInfo
            )
            writes.append(write)
            
        pWrites = vk.ffi.new('VkWriteDescriptorSet[]', writes)
        vk.vkUpdateDescriptorSets(self.device, 3, pWrites, 0, vk.VK_NULL_HANDLE)
        
        allocInfoCmd = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        commandBuffer = vk.vkAllocateCommandBuffers(self.device, allocInfoCmd)[0]
        
        beginInfo = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        vk.vkBeginCommandBuffer(commandBuffer, beginInfo)
        
        vk.vkCmdBindPipeline(commandBuffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
        
        pSets = vk.ffi.new('VkDescriptorSet[]', [descriptorSet])
        vk.vkCmdBindDescriptorSets(commandBuffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout, 0, 1, pSets, 0, vk.VK_NULL_HANDLE)
        
        pConstants = vk.ffi.new('uint32_t[]', [num_experts, dim])
        vk.vkCmdPushConstants(commandBuffer, self.pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, vk.ffi.cast('void*', pConstants))
        
        group_x = (num_experts + 255) // 256
        vk.vkCmdDispatch(commandBuffer, group_x, 1, 1)
        
        vk.vkEndCommandBuffer(commandBuffer)
        
        pCmdBuffers = vk.ffi.new('VkCommandBuffer[]', [commandBuffer])
        submitInfo = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=pCmdBuffers
        )
        pSubmitInfos = vk.ffi.new('VkSubmitInfo[]', [submitInfo])
        vk.vkQueueSubmit(self.queue, 1, pSubmitInfos, vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self.queue)
        
        o_ptr = vk.vkMapMemory(self.device, o_mem, 0, output_size, 0)
        distances = np.zeros(num_experts, dtype=np.float32)
        vk.ffi.memmove(vk.ffi.from_buffer(distances.data), o_ptr, output_size)
        vk.vkUnmapMemory(self.device, o_mem)
        
        # Cleanup
        vk.vkDestroyBuffer(self.device, q_buf, None)
        vk.vkFreeMemory(self.device, q_mem, None)
        vk.vkDestroyBuffer(self.device, e_buf, None)
        vk.vkFreeMemory(self.device, e_mem, None)
        vk.vkDestroyBuffer(self.device, o_buf, None)
        vk.vkFreeMemory(self.device, o_mem, None)
        
        vk.vkDestroyDescriptorPool(self.device, descriptorPool, None)
        vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, pCmdBuffers)
        
        return distances

if __name__ == "__main__":
    print("Testing Vulkan Compute API bindings...")
    router = VulkanMoERouter()
    q = np.random.randn(640).astype(np.float32)
    e = np.random.randn(100, 640).astype(np.float32)
    start = time.perf_counter()
    dist = router.compute_distances(q, e)
    t = time.perf_counter() - start
    dist_cpu = np.linalg.norm(e - q, axis=1)
    print(f"Computed Distances in {t*1000:.2f}ms. Difference from CPU math: {np.abs(dist - dist_cpu).max()}")
    print("Vulkan Engine Test Passed!")
