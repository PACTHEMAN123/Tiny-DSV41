import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {webcrypto} from "node:crypto";
import {performance} from "node:perf_hooks";
import test from "node:test";
import vm from "node:vm";

const html=readFileSync(new URL("../../rollout_manager/terminal_bench/browser_relay/relay.html",import.meta.url),"utf8");
const source=html.slice(html.indexOf('"use strict";'),html.indexOf('document.querySelector("#connect").onclick'));
const id="1".repeat(32),job={id,operation:"generate",payload:{model:"qwen"},deadline_unix:Date.now()/1000+60};

function browser(role,result) {
  const sent=[],commits=[];
  const context=vm.createContext({URL,URLSearchParams,AbortController,AbortSignal,TextEncoder,TextDecoder,
    performance,crypto:webcrypto,setTimeout,clearTimeout,addEventListener(){},
    location:{hash:"#token="+"a".repeat(64),pathname:"/proxy/18093/relay.html",search:"",
      href:"https://"+role+".test/proxy/18093/relay.html",origin:"https://"+role+".test"},
    history:{replaceState(){}},document:{querySelector(){return {textContent:"",disabled:false};}},
    recordCommit(path,value){if(path==="api/result")commits.push(JSON.parse(JSON.stringify(value)));},
    recordPost(_peer,_origin,message){
      sent.push(JSON.parse(JSON.stringify(message)));
      if(role==="gpu" && message.type==="error")vm.runInContext('acknowledgeCompletion("'+id+'")',context);
      if(role==="cpu" && message.type==="request") {
        queueMicrotask(()=>vm.runInContext('receiveMessage({origin:config.peer_origin,source:peerWindow,data:{...fixtureResult,type:"error",nonce}})',context));
      }
      if(role==="cpu" && message.type==="complete_ack") {
        vm.runInContext('receiveMessage({origin:config.peer_origin,source:peerWindow,data:{type:"credit_released",id:"'+id+'",nonce}})',context);
      }
    }});
  vm.runInContext(source,context);
  vm.runInContext('config={role:'+JSON.stringify(role)+',serving_profile:"vllm-training-v1",model:"qwen",boot_id:"a".repeat(32),max_inflight:1,generation_seconds:360,peer_origin:"https://peer.test"};paired=true;peerCapacity=1;peerGenerationSeconds=360;peerWindow={closed:false};nonce="b".repeat(64);telemetry=async()=>{};post=recordPost;api=async(path,value)=>{recordCommit(path,value);return {acknowledged:true};};phaseApi=async(phase,path,value)=>path==="api/submit"?{id:value.id,admitted:true}:{acknowledged:true};',context);
  vm.runInContext('fixtureResult='+JSON.stringify(result)+';waitCompletion=async()=>fixtureResult;',context);
  return {run:s=>vm.runInContext(s,context),sent,commits};
}

for(const outcome of ["completed","unknown",null]) {
  test("both browsers preserve "+(outcome||"untyped")+" negative without deriving sampler closure",async()=>{
    const receipt={schema_version:1,request_id:id,request_payload_sha256:"a".repeat(64),
      policy_ack:{version:0,digest:"b".repeat(64),lineage:"fixture"},stage:"fixture",
      sampler_outcome:outcome,sampler_drained:outcome==="completed",actual_actor_drain_proven:outcome==="completed",
      drain_scope:"request",proof_kind:outcome==="completed"?"owned_vllm_final_request_output":"none",automatic_replay:false};
    const negative={id,pending:false,ok:false,error:"original actor failure",...(outcome?{generation_failure:receipt}:{})};
    const gpu=browser("gpu",negative);
    await gpu.run('gpuRequest('+JSON.stringify(job)+',10000)');
    await gpu.run('Promise.all([...gpuTasks.values()].map(item=>item.cleanupPromise))');
    const message=gpu.sent.find(value=>value.type==="error");
    assert.equal(message.error,negative.error);
    if(outcome)assert.deepEqual(message.generation_failure,receipt);
    else assert.ok(!Object.hasOwn(message,"generation_failure"));
    assert.equal(gpu.run("gpuTasks.size"),0);
    const cpu=browser("cpu",message);
    await assert.rejects(cpu.run('processJob('+JSON.stringify(job)+',10000)'),/original actor failure/);
    assert.equal(cpu.commits.length,1,"one durable negative commit, no generation replay");
    assert.deepEqual(cpu.commits[0],{id,ok:false,error:negative.error,...(outcome?{generation_failure:receipt}:{})});
    assert.equal(cpu.run("cpuTasks.size"),0);
  });
}
