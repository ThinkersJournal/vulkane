// SPDX-License-Identifier: MIT OR Apache-2.0
//! Graphics-pipeline binding, draws and dynamic state on [`CommandBufferRecording`].

use super::*;

impl<'a> CommandBufferRecording<'a> {
    /// Record `vkCmdBindPipeline` for a graphics pipeline.
    pub fn bind_graphics_pipeline(&mut self, pipeline: &GraphicsPipeline) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBindPipeline
            .expect("vkCmdBindPipeline is required by Vulkan 1.0");
        // Safety: command buffer is recording, pipeline handle is valid.
        unsafe {
            cmd(
                self.buffer.handle,
                VkPipelineBindPoint::PIPELINE_BIND_POINT_GRAPHICS,
                pipeline.handle,
            )
        };
    }

    /// Record `vkCmdBindDescriptorSets` for a graphics pipeline.
    pub fn bind_graphics_descriptor_sets(
        &mut self,
        layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[&DescriptorSet],
    ) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBindDescriptorSets
            .expect("vkCmdBindDescriptorSets is required by Vulkan 1.0");

        let raw: Vec<VkDescriptorSet> = descriptor_sets.iter().map(|s| s.handle).collect();
        // Safety: command buffer is recording, raw lives until call end.
        unsafe {
            cmd(
                self.buffer.handle,
                VkPipelineBindPoint::PIPELINE_BIND_POINT_GRAPHICS,
                layout.handle,
                first_set,
                raw.len() as u32,
                raw.as_ptr(),
                0,
                std::ptr::null(),
            )
        };
    }

    /// Record `vkCmdBindVertexBuffers`. Each entry is a `(buffer, offset)`
    /// tuple. The first entry is bound to slot `first_binding`, the
    /// next to `first_binding + 1`, etc.
    pub fn bind_vertex_buffers(&mut self, first_binding: u32, buffers_offsets: &[(&Buffer, u64)]) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBindVertexBuffers
            .expect("vkCmdBindVertexBuffers is required by Vulkan 1.0");

        let raw_buffers: Vec<VkBuffer> = buffers_offsets.iter().map(|(b, _)| b.handle).collect();
        let raw_offsets: Vec<u64> = buffers_offsets.iter().map(|(_, o)| *o).collect();
        // Safety: command buffer is recording, raw_* outlive the call.
        unsafe {
            cmd(
                self.buffer.handle,
                first_binding,
                raw_buffers.len() as u32,
                raw_buffers.as_ptr(),
                raw_offsets.as_ptr(),
            )
        };
    }

    /// Record `vkCmdBindIndexBuffer`. `index_type` is `VK_INDEX_TYPE_UINT16` (0)
    /// or `VK_INDEX_TYPE_UINT32` (1).
    pub fn bind_index_buffer(&mut self, buffer: &Buffer, offset: u64, index_type: u32) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBindIndexBuffer
            .expect("vkCmdBindIndexBuffer is required by Vulkan 1.0");
        // Safety: command buffer is recording, buffer handle is valid.
        let it = match index_type {
            0 => VkIndexType::INDEX_TYPE_UINT16,
            _ => VkIndexType::INDEX_TYPE_UINT32,
        };
        unsafe { cmd(self.buffer.handle, buffer.handle, offset, it) };
    }

    /// Record `vkCmdDraw`: emit `vertex_count` vertices starting at
    /// `first_vertex`, repeated for `instance_count` instances starting
    /// at `first_instance`.
    pub fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdDraw
            .expect("vkCmdDraw is required by Vulkan 1.0");
        // Safety: command buffer is recording, pipeline must be bound.
        unsafe {
            cmd(
                self.buffer.handle,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            )
        };
    }

    /// Record `vkCmdDrawIndexed`.
    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdDrawIndexed
            .expect("vkCmdDrawIndexed is required by Vulkan 1.0");
        // Safety: command buffer is recording, pipeline + index buffer must be bound.
        unsafe {
            cmd(
                self.buffer.handle,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        };
    }

    /// Record `vkCmdSetViewport`. Used when the pipeline was built with
    /// `VK_DYNAMIC_STATE_VIEWPORT`.
    pub fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdSetViewport
            .expect("vkCmdSetViewport is required by Vulkan 1.0");
        let viewport = VkViewport {
            x,
            y,
            width,
            height,
            minDepth: 0.0,
            maxDepth: 1.0,
        };
        // Safety: command buffer is recording, viewport lives until call end.
        unsafe { cmd(self.buffer.handle, 0, 1, &viewport) };
    }

    /// Record `vkCmdSetScissor`.
    pub fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdSetScissor
            .expect("vkCmdSetScissor is required by Vulkan 1.0");
        let scissor = VkRect2D {
            offset: VkOffset2D { x, y },
            extent: VkExtent2D { width, height },
        };
        // Safety: command buffer is recording, scissor lives until call end.
        unsafe { cmd(self.buffer.handle, 0, 1, &scissor) };
    }
}
