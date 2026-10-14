import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {webcrypto} from "node:crypto";
import {performance} from "node:perf_hooks";
import test from "node:test";
import vm from "node:vm";

function browser() {
  const html=readFileSync(new URL("../../rollout_manager/terminal_bench/browser_relay/relay.html",import.meta.url),"utf8");
  const source=html.slice(html.indexOf('"use strict";'),html.indexOf('document.querySelector("#connect").onclick'));
  const context=vm.createContext({URL,URLSearchParams,AbortController,AbortSignal,TextEncoder,TextDecoder,
    performance,crypto:webcrypto,setTimeout,clearTimeout,addEventListener(){},
    location:{hash:"#token="+"a".repeat(64),pathname:"/proxy/18093/relay.html",search:"",
      href:"https://gpu.test/proxy/18093/relay.html",origin:"https://gpu.test"},
    history:{replaceState(){}},document:{querySelector(){return {textContent:"",disabled:false};}}});
  vm.runInContext(source,context);
  vm.runInContext('config={role:"gpu",serving_profile:"vllm-training-v1",model:"qwen"};',context);
  return {call:(fn,v)=>{context.value=structuredClone(v);return JSON.parse(JSON.stringify(vm.runInContext(fn+"(value)",context)));}};
}
function fixture() {
  const ack={version:0,digest:"a".repeat(64),lineage:"fixture"};
  const kernel={moe_backend:"triton",enable_jit_warmup:false,enable_cutedsl_warmup:false,enable_flashinfer_autotune:false};
  const adapter={profile:"qwen3-moe-minimax-import-guard-v1",applied_count:1,stock_kernel_warmup_sha256:"69d9f633e353d02898b0e0054613f10596747341ea23e4369c1cdfd88ec2cac5",model_type:"qwen3_moe",preserves_original_warmup_body:true,bypassed_unrelated_minimax_import:true};
  const moe=[{module:"model.layers.0.mlp.experts",backend:"TRITON",selected_experts_class:"vllm.model_executor.layers.fused_moe.experts.triton_moe.TritonExperts",experts_class:"vllm.model_executor.layers.fused_moe.experts.triton_moe.TritonExperts"}];
  return {role:"gpu",serving_profile:"vllm-training-v1",max_inflight:1,training_controller:{
    schema_version:1,serving_profile:"vllm-training-v1",served_model:"qwen",generation_slots:1,
    controller_state:"running",active_generations:0,model_pid:10,model_pid_start_ticks:20,
    netns_inode:30,model_cmdline_sha256:"b".repeat(64),started_unix:1000,expires_unix:13000,
    observed_unix:1001,model_dir:"/mnt/fuse/qwen",python_executable:"/venv/bin/python",
    policy:{...ack,identities:{rollout_engine:"vllm",vllm_max_model_len:"32768",vllm_kv_cache_memory_bytes:"34359738368",vllm_moe_backend:"triton",vllm_enable_jit_warmup:"false",vllm_enable_cutedsl_warmup:"false",vllm_enable_flashinfer_autotune:"false",vllm_startup_adapter:"qwen3-moe-minimax-import-guard-v1",vllm_worker_rpc:"rollout_manager_owned_call",vllm_insecure_serialization:"false"},sampling:{temperature:1}},policy_ack:ack,
    capabilities:{schema_version:1},vllm_engine:{schema_version:1,version:"0.31.0",serving_profile:"vllm-training-v1",
      tensor_parallel_size:1,logprobs_mode:"processed_logprobs",prefix_caching:false,enforce_eager:false,
      compilation_mode:0,cudagraph_mode:"FULL_DECODE_ONLY",cudagraph_capture_sizes:[1],loaded_policy_ack:{...ack},loaded_trainable_tensors:{sha256:"c".repeat(64)},
      max_model_len:32768,kv_cache_memory_bytes:34359738368,gpu_memory_utilization:.70,kv_bytes_per_token:98304,full_context_raw_kv_bytes:3221225472,kernel_config:kernel,startup_adapter:{...adapter},worker_rpc_method:"rollout_manager_owned_call",insecure_serialization:false,actor_process:{pid:39,start_ticks:40,cmdline_sha256:"d".repeat(64)},workers:[{pid:40,start_ticks:41,cmdline_sha256:"e".repeat(64),device:"cuda:0",device_uuid:"GPU-fixture",kernel_config:{...kernel},startup_adapter:{...adapter},worker_rpc_method:"rollout_manager_owned_call",insecure_serialization:false,moe_implementations:moe,graph:{captured_graph_count:1,replay_calls:3}}]}}};
}
test("vLLM proof preserves distinct profile and pins engine owners across policy versions",()=>{
  const b=browser(),v=fixture();
  assert.deepEqual(b.call("modelCapacityProof",v),{training_controller:v.training_controller});
  const before=b.call("modelCapacityIdentity",v),updated=structuredClone(v);
  updated.training_controller.policy.version=1;
  updated.training_controller.policy_ack.version=1;
  updated.training_controller.vllm_engine.loaded_policy_ack.version=1;
  updated.training_controller.vllm_engine.workers[0].graph.replay_calls=4;
  assert.deepEqual(b.call("modelCapacityIdentity",updated),before);
  updated.training_controller.vllm_engine.workers[0].pid=41;
  assert.notDeepEqual(b.call("modelCapacityIdentity",updated),before);
});
test("vLLM proof rejects native alias, legacy capacity, raw logprobs and stale loaded ACK",()=>{
  const b=browser();
  for(const mutate of [v=>{v.native_controller=v.training_controller;delete v.training_controller;},
    v=>{v.model_capacity_source="owned_vllm_process";},
    v=>{v.training_controller.vllm_engine.logprobs_mode="raw_logprobs";},
    v=>{v.training_controller.vllm_engine.loaded_policy_ack.version=1;},
    v=>{v.training_controller.vllm_engine.enforce_eager=true;},
    v=>{v.training_controller.vllm_engine.gpu_memory_utilization=.60;},
    v=>{v.training_controller.vllm_engine.full_context_raw_kv_bytes=1;},
    v=>{delete v.training_controller.vllm_engine.actor_process.start_ticks;},
    v=>{v.training_controller.vllm_engine.kernel_config.enable_jit_warmup=true;},
    v=>{v.training_controller.policy.identities.vllm_moe_backend="auto";},
    v=>{v.training_controller.vllm_engine.workers[0].kernel_config.moe_backend="auto";},
    v=>{v.training_controller.vllm_engine.workers[0].moe_implementations[0].experts_class="other";},
    v=>{v.training_controller.vllm_engine.startup_adapter.preserves_original_warmup_body=false;},
    v=>{v.training_controller.vllm_engine.workers[0].startup_adapter.applied_count=2;},
    v=>{v.training_controller.vllm_engine.insecure_serialization=true;},
    v=>{v.training_controller.vllm_engine.workers[0].worker_rpc_method="callable";},
    v=>{delete v.training_controller;}]) {
    const v=fixture();mutate(v);assert.throws(()=>b.call("modelCapacityProof",v));
  }
});
