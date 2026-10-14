import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {webcrypto} from "node:crypto";
import {performance} from "node:perf_hooks";
import vm from "node:vm";
import test from "node:test";

const html=readFileSync(new URL("../../rollout_manager/terminal_bench/browser_relay/relay.html",import.meta.url),"utf8");
const source=html.slice(html.indexOf('"use strict";'),html.indexOf('document.querySelector("#connect").onclick'));
function browser(role,options={}) {
  const calls=[],sent=[],nodes=new Map();
  const node=selector=>{if(!nodes.has(selector))nodes.set(selector,{textContent:"",disabled:false});return nodes.get(selector);};
  const context=vm.createContext({URL,URLSearchParams,AbortController,AbortSignal,TextEncoder,TextDecoder,
    crypto:webcrypto,performance:options.performance||performance,setTimeout:options.setTimeout||setTimeout,clearTimeout:options.clearTimeout||clearTimeout,addEventListener(){},
    location:{hash:"#token="+"a".repeat(64),pathname:"/relay.html",href:"https://"+role+".test/relay.html",origin:"https://"+role+".test"},
    history:{replaceState(){}},document:{querySelector:node},
    record(path,value){calls.push({path,value});return {id:value.id,durable:true};},
    send(_peer,_origin,value){sent.push(value);options.onSend?.(value);}});
  vm.runInContext(source,context);
  vm.runInContext('config={role:'+JSON.stringify(role)+',boot_id:"'+(role==="cpu"?"1":"2").repeat(32)+'",peer_origin:"https://peer.test",model:"qwen",max_inflight:1,generation_seconds:360,serving_profile:"native-pytorch-v1",artifact_transport:"durable-chunks-v1"};peerBootId="'+(role==="cpu"?"2":"1").repeat(32)+'";paired=true;peerWindow={closed:false};peerCapacity=1;peerGenerationSeconds=360;nonce="a".repeat(64);relayDeadline=performance.now()+120000;originalApi=api;api=async(path,value)=>record(path,value);post=send;',context);
  return {context,calls,sent,node,run:code=>vm.runInContext(code,context),receive:value=>{context.fixture=value;vm.runInContext('receiveMessage({origin:config.peer_origin,source:peerWindow,data:{...fixture,nonce}})',context);}};
}
test("artifact route preserves authenticated peer identity and durable commit response",async()=>{
  const gpu=browser("gpu"),request={type:"artifact_request",rpc_id:"3".repeat(32),boot_id:"1".repeat(32),
    route:"api/artifact-commit",value:{id:"4".repeat(64)},nonce:"a".repeat(64)};
  gpu.run('fixture='+JSON.stringify(request));
  gpu.run('receiveMessage({origin:"https://evil.test",source:peerWindow,data:fixture})');
  gpu.run('receiveMessage({origin:config.peer_origin,source:{},data:fixture})');
  assert.equal(gpu.calls.length,0);
  gpu.run('receiveMessage({origin:config.peer_origin,source:peerWindow,data:{...fixture,nonce:"wrong"}})');
  assert.equal(gpu.calls.length,0);
  gpu.run('receiveMessage({origin:config.peer_origin,source:peerWindow,data:fixture})');
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(gpu.calls.length,1);
  assert.equal(gpu.calls[0].path,"api/artifact-commit");
  assert.equal(gpu.sent[0].ok,true);
  assert.equal(gpu.sent[0].receipt.durable,true);
  assert.equal(gpu.sent[0].rpc_id,request.rpc_id);
  assert.equal(gpu.sent[0].boot_id,"2".repeat(32));
});
test("artifact peer cannot invoke generation or arbitrary file routes",async()=>{
  const gpu=browser("gpu");
  await gpu.run('artifactReceive({rpc_id:"3".repeat(32),boot_id:peerBootId,route:"api/submit",value:{id:"4".repeat(32)}})');
  assert.equal(gpu.calls.length,0);
  assert.equal(gpu.sent.length,0);
});

function nativeConfig(version=0) {
  const ack={version,digest:String(version+1).repeat(64),lineage:"fixture"};
  return {role:"gpu",boot_id:"2".repeat(32),model:"qwen",max_inflight:1,generation_seconds:360,
    serving_profile:"native-pytorch-v1",uncertain:0,lifetime_remaining_ms:120000,
    training_controller:undefined,native_controller:{schema_version:1,serving_profile:"native-pytorch-v1",served_model:"qwen",generation_slots:1,controller_state:"running",active_generations:0,
      model_pid:10,model_pid_start_ticks:20,netns_inode:30,model_cmdline_sha256:"b".repeat(64),started_unix:1000,expires_unix:13000,observed_unix:1001,
      model_dir:"/mounted/qwen",python_executable:"/venv/bin/python",policy:{...ack,identities:{tokenizer_sha256:"c".repeat(64)},sampling:{temperature:.8}},policy_ack:ack,capabilities:{schema_version:1}}};
}
function gateway503() {
  return {status:503,ok:false,redirected:false,url:"https://gpu.test/api/config",headers:{get:()=>"text/html"},text:async()=>"Gateway unavailable".padEnd(808," ")};
}
test("only exact transient config503 retains pairing; identity and authentication failures invalidate it",async()=>{
  const gpu=browser("gpu");
  const initial=nativeConfig();delete initial.training_controller;
  gpu.context.initial=initial;
  gpu.run('config={...config,...initial};api=originalApi');gpu.context.fetch=async()=>gateway503();
  await gpu.run('capacityProbeRequest("3".repeat(32))');
  assert.equal(gpu.run('paired'),true);
  assert.equal(gpu.sent.at(-1).type,"capacity_unavailable");
  assert.equal(gpu.sent.at(-1).reason,"temporary_config_503");
  for(const mutate of [r=>{r.status=401;},r=>{r.redirected=true;},r=>{r.url="https://other.test/api/config";},r=>{r.url="https://gpu.test/api/status";}]) {
    const response=gateway503();mutate(response);gpu.context.fetch=async()=>response;gpu.run('paired=true');
    await gpu.run('capacityProbeRequest("3".repeat(32))');assert.equal(gpu.run('paired'),false);
  }
  const changed=nativeConfig(1);delete changed.training_controller;changed.native_controller.model_pid=11;
  gpu.context.fetch=async()=>({...gateway503(),status:200,ok:true,headers:{get:()=>"application/json"},text:async()=>JSON.stringify(changed)});
  gpu.run('paired=true');await gpu.run('capacityProbeRequest("3".repeat(32))');
  assert.equal(gpu.run('paired'),false);assert.equal(gpu.sent.at(-1).type,"capacity_invalid");
});
test("lease-v1 artifact commits while generation readiness is paused after config503",async()=>{
  const gpu=browser("gpu",{setTimeout:fn=>setImmediate(fn),clearTimeout:clearImmediate});
  gpu.context.fetch=async()=>gateway503();gpu.run('api=originalApi');
  await gpu.run('capacityProbeRequest("3".repeat(32))');assert.equal(gpu.run('paired'),true);
  const manifest={id:"4".repeat(64),kind:"policy_lease",key:"lease-v1-evaluation",bytes:3433,sha256:"5".repeat(64),chunk_bytes:131072};
  const delivered=[];let acknowledged=false;
  gpu.context.localApi=async(route,value)=>{
    if(route==="api/artifact-next")return manifest;
    if(route==="api/artifact-read")return {id:manifest.id,index:0,sha256:"6".repeat(64),data:"chunk"};
    if(route==="api/artifact-ack"){acknowledged=true;gpu.run('paired=false');return {acknowledged:true};}
    throw new Error("Unexpected model/admission route "+route);
  };
  gpu.context.peerArtifact=async(route,value)=>{
    assert.equal(gpu.run('paired'),true);delivered.push(route);
    if(route==="api/artifact-start")return {id:manifest.id,next_index:0};
    if(route==="api/artifact-chunk")return {id:manifest.id,index:0,sha256:value.sha256,durable:true};
    return {...manifest,durable:true};
  };
  gpu.run('api=localApi;artifactRPC=peerArtifact');await gpu.run('artifactPump()');
  assert.deepEqual(delivered,["api/artifact-start","api/artifact-chunk","api/artifact-commit"]);assert.equal(acknowledged,true);
});
test("heartbeat survives more than three expired probes and recovers exact v1 without admitting work",async()=>{
  let now=0,probes=0,cpu;const observations=[],delays=[];
  const timer=(fn,ms)=>setImmediate(()=>{now+=ms;delays.push(ms);fn();});
  cpu=browser("cpu",{performance:{now:()=>now},setTimeout:timer,clearTimeout:clearImmediate,onSend:value=>{
    if(value.type!=="capacity_probe")return;
    probes++;
    if(probes===5)cpu.receive({type:"capacity_unavailable",id:value.id,boot_id:"2".repeat(32),reason:"temporary_config_503"});
    if(probes===6){const ready=nativeConfig(1);delete ready.training_controller;cpu.receive({type:"capacity",id:value.id,protocol:2,...ready});}
  }});
  cpu.context.localApi=async(route,value)=>{
    if(route==="api/pulse")return {challenge_id:String(probes+1).padStart(32,"0"),ttl_ms:15000};
    assert.equal(route,"api/pair");assert.equal(value.native_controller.policy_ack.version,1);return {recorded:true};
  };
  cpu.context.observe=()=>{observations.push({fresh:cpu.run('capacityFreshUntil'),paired:cpu.run('paired')});if(cpu.run('capacityFreshUntil')>now)cpu.run('heartbeatEnabled=false');};
  cpu.run('api=localApi;originalWake=wakeRelay;wakeRelay=()=>{observe();originalWake();}');
  await cpu.run('heartbeat()');
  assert.equal(probes,6);assert.equal(observations.filter(x=>x.fresh>0).length,1);assert.equal(observations.every(x=>x.paired),true);
  assert.deepEqual(delays.filter(ms=>ms<15000),[1000,2000,4000,5000,5000]);
  assert.equal(cpu.run('running'),false);assert.equal(cpu.calls.length,0);
});
test("stop cancels readiness recovery and expired sessions cannot resume admission",async()=>{
  for(const control of ["#stop","#cancel"]) {
    let cpu;const routes=[];
    cpu=browser("cpu",{onSend:value=>{if(value.type==="capacity_probe")cpu.node(control).onclick();}});
    const controls=html.slice(html.indexOf('document.querySelector("#connect").onclick'),html.indexOf('\n(async()=>{'));
    vm.runInContext(controls,cpu.context);
    cpu.context.localApi=async route=>{routes.push(route);return {challenge_id:"3".repeat(32),ttl_ms:15000};};cpu.run('api=localApi;running=true');
    await cpu.run('heartbeat()');
    assert.equal(cpu.run('heartbeatEnabled'),false);assert.equal(cpu.run('heartbeatActive'),false);assert.equal(cpu.run('running'),false);
    await cpu.run('heartbeat()');assert.equal(cpu.sent.filter(v=>v.type==="capacity_probe").length,1);
    cpu.run('staleProbe={id:"3".repeat(32),started:performance.now(),epoch:heartbeatEpoch-1};capacityProbe=staleProbe');
    cpu.context.ready=nativeConfig(1);delete cpu.context.ready.training_controller;
    await assert.rejects(cpu.run('recordCapacity(ready,staleProbe)'),/superseded/);
    assert.deepEqual(routes,["api/pulse"]);
    cpu.run('capacityProbe=null;heartbeatEnabled=true;paired=true;relayDeadline=0');await cpu.run('heartbeat()');
    assert.equal(cpu.run('paired'),false);assert.equal(cpu.run('capacityFreshUntil'),0);
  }
});
