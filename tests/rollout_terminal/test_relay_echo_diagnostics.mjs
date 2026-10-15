import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {webcrypto} from "node:crypto";
import {performance} from "node:perf_hooks";
import test from "node:test";
import vm from "node:vm";

const secret="PRIVATE_GATEWAY_BODY_MUST_NOT_APPEAR";
function browser(response) {
  const html=readFileSync(new URL("../../rollout_manager/terminal_bench/browser_relay/relay.html",import.meta.url),"utf8");
  const source=html.slice(html.indexOf('"use strict";'),html.indexOf('document.querySelector("#connect").onclick'));
  const sent=[],calls=[];
  const context=vm.createContext({URL,URLSearchParams,AbortController,AbortSignal,TextEncoder,TextDecoder,
    performance,crypto:webcrypto,setTimeout,clearTimeout,addEventListener(){},
    location:{hash:"#token="+"a".repeat(64),pathname:"/proxy/18093/relay.html",search:"",
      href:"https://gpu.test/proxy/18093/relay.html",origin:"https://gpu.test"},
    history:{replaceState(){}},document:{querySelector(){return {textContent:"",disabled:false};}},
    fetch:async(url,options)=>{calls.push({url:String(url),options});return response;},
    recordPost(_peer,_origin,message){
      sent.push(JSON.parse(JSON.stringify(message)));
      if(message.type==="error") {
        context.deliveredId=message.id;
        vm.runInContext("acknowledgeCompletion(deliveredId)",context);
      }
    }});
  vm.runInContext(source,context);
  vm.runInContext('config={role:"gpu",serving_profile:"native-pytorch-v1",model:"qwen",boot_id:"a".repeat(32),max_inflight:1,generation_seconds:360,peer_origin:"https://cpu.test"};paired=true;peerCapacity=1;peerWindow={closed:false};nonce="b".repeat(64);api=async()=>({});telemetry=async()=>{};post=recordPost;',context);
  return {run:s=>vm.runInContext(s,context),sent,calls};
}

for(const [label,status,type,body] of [["rejected origin",403,"application/json",{}],
    ["gateway HTML",502,"text/html",{}],["missing stream",204,"application/x-ndjson",null]]) {
  test("echo diagnostic identifies "+label+" without exposing private response data",async()=>{
    const response={status,ok:status>=200&&status<300,body,redirected:false,
      url:"https://gpu.test/proxy/18093/api/echo?private="+secret,
      headers:{get:name=>name==="Content-Type"?type:null},text:async()=>secret};
    const b=browser(response);
    await b.run('gpuRequest({id:"1".repeat(32),operation:"echo",payload:{chunks:3},deadline_unix:Date.now()/1000+120},120000)');
    const error=b.sent.find(message=>message.type==="error").error;
    assert.match(error,/Echo stream unavailable/);
    assert.ok(!error.includes(secret));
    const diagnostic=JSON.parse(error.match(/http_diagnostic=(\{.*\})\]/)[1]);
    assert.equal(diagnostic.http_status,status);
    assert.equal(diagnostic.request_route,"api/echo");
    assert.equal(diagnostic.final_origin_category,"same_origin");
    assert.equal(diagnostic.final_path_category,"relay_api");
    assert.equal(b.calls.length,1,"echo admission is never retried");
    assert.equal(b.calls[0].options.method,"POST");
    assert.equal(b.calls[0].options.headers["X-Relay-Origin"],"https://gpu.test");
    assert.equal(b.run("gpuTasks.size"),0,"CPU error commit releases only browser qualification credit");
  });
}
