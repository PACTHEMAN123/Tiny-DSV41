import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.runtime import train_groups
from rollout_manager.benchmark_training import artifact_policy
from rollout_manager.terminal_bench.preupdate_recovery import original_initial_policy, HistoricalGroupDelivery
from rollout_manager.training import PolicyRef, TrainingGroup, TrainingTrajectory, TurnTrace, token_digest


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head=torch.nn.Linear(3,6,bias=False)
        self.config=SimpleNamespace(model_type='tiny',vocab_size=6,max_position_embeddings=32)
        self.calls=0
    def forward(self,ids,attention_mask=None):
        self.calls+=1
        x=torch.nn.functional.one_hot(ids%3,3).float()
        return SimpleNamespace(logits=self.lm_head(x))
    def assert_engram_invariants(self): pass


def group(rewards=(0.,0.),policy=None):
    policy=policy or PolicyRef(0,'a'*64,'original',sampling={'temperature':.8,'top_p':1,'top_k':0})
    samples=[TrainingTrajectory(0,'task',token_digest([1,2]),'b'*64,i,
        [TurnTrace(1,str(i),[1,2],[2,3],[-1.,-2.],policy)],reward)
        for i,reward in enumerate(rewards)]
    return TrainingGroup(0,'task',token_digest([1,2]),'b'*64,policy,len(samples),tuple(samples))


class ZeroObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.config=RLConfig(trainable_scope='lm_head',temperature=.8,kl_beta=0,weight_decay=0,rollout_engine='native-pytorch')
    def test_fresh_optimizer_really_steps_without_replay_or_weight_change(self):
        model=Head();before=model.lm_head.weight.detach().clone();opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0)
        result=train_groups(model,None,opt,[group()],self.config)
        self.assertEqual(model.calls,0);self.assertTrue(result['replay_skipped']);self.assertTrue(result['optimizer_step_applied'])
        self.assertEqual(result['grad_norm'],0);self.assertEqual(result['backward_calls'],0)
        self.assertNotIn('importance_ratio',result);self.assertNotIn('clip_fraction',result)
        self.assertTrue(torch.equal(before,model.lm_head.weight));self.assertEqual(float(opt.state[model.lm_head.weight]['step']),1.)
    def test_existing_adam_momentum_matches_real_zero_backward(self):
        model=Head();opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0)
        model.lm_head.weight.grad=torch.ones_like(model.lm_head.weight);opt.step()
        reference=copy.deepcopy(model);other=torch.optim.AdamW(reference.parameters(),lr=.01,weight_decay=0);other.load_state_dict(copy.deepcopy(opt.state_dict()))
        train_groups(model,None,opt,[group()],self.config)
        with patch('dsv41_train.rl.runtime._exact_zero_objective',return_value=False):
            train_groups(reference,None,other,[group()],self.config)
        self.assertTrue(torch.equal(model.lm_head.weight,reference.lm_head.weight))
        for key,value in opt.state[model.lm_head.weight].items():self.assertTrue(torch.equal(value,other.state[reference.lm_head.weight][key]))
        self.assertGreater(reference.calls,0)
    def test_nonzero_advantages_and_kl_keep_real_replay(self):
        model=Head();opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0)
        result=train_groups(model,None,opt,[group((0.,1.))],self.config)
        self.assertGreater(model.calls,0);self.assertFalse(result.get('replay_skipped',False))
        model=Head();ref=copy.deepcopy(model);opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0)
        cfg=replace(self.config,kl_beta=.1)
        train_groups(model,ref,opt,[group()],cfg);self.assertGreater(model.calls,0)
    def test_zero_path_rejects_mismatched_temperature_tokens_or_optimizer(self):
        for mode in ('temperature','token','weight_decay','nonfinite'):
            model=Head();opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01 if mode=='weight_decay' else 0)
            item=group()
            if mode=='temperature':
                item=group(policy=PolicyRef(0,'a'*64,'original',sampling={'temperature':.9,'top_p':1,'top_k':0}))
            elif mode=='token':
                object.__setattr__(item.samples[0].turns[0],'action_token_ids',(6,3))
            elif mode=='nonfinite':
                with torch.no_grad():model.lm_head.weight[0,0]=float('nan')
            with self.subTest(mode=mode),self.assertRaises((ValueError,FloatingPointError)):
                train_groups(model,None,opt,[item],self.config)
            self.assertEqual(opt.state,{})


class AutomaticProviderTests(unittest.TestCase):
    def test_automatic_initial_seal_is_received_before_waiting_for_manifest(self):
        from rollout_manager.terminal_bench.native_workflow import SealedCPUGroupProvider
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();output=root/'training';output.mkdir();manifest=root/'incoming/episodes/group-0.seal.json';item=group();calls=[]
            provider=SealedCPUGroupProvider.__new__(SealedCPUGroupProvider)
            provider._error=None;provider._paused=False;provider._next=0;provider._retained={};provider.manifests=(manifest,)
            provider._historical_delivery=SimpleNamespace(validate=lambda value:{'delivery_drain_complete':True})
            provider._automatic_artifacts=True;provider._initial_lease_published=False;provider.controller=object()
            provider._relay_root=root/'relay';provider._relay_port=18095;provider.config=SimpleNamespace(group_size=2,worker_timeout_seconds=1)
            provider.output=output;provider._expected_original_seal_sha256=None;provider.task_ids=('task',)
            provider._expected_max_turns=None
            provider.native=SimpleNamespace(policy=item.policy);provider._ack={k:getattr(item.policy,k) for k in ('version','digest','lineage')}
            provider.lease={'tasks':{'task':{'initial_context_sha256':item.initial_context_sha256,'reset_sha256':item.reset_sha256}}}
            def publish(relay,kind,key,path):calls.append(('publish',kind,key))
            def receive(relay,key,directory,timeout):
                calls.append(('receive',key));directory.mkdir(parents=True);manifest.write_text('{}');return manifest
            with patch('rollout_manager.terminal_bench.artifact_transport.publish_file',publish),patch('rollout_manager.terminal_bench.artifact_transport.receive_sealed',receive),patch('rollout_manager.terminal_bench.sealed.validate_sealed_group',return_value=item):
                packet=provider.get(.2)
            self.assertEqual(calls,[('publish','policy_lease','lease-v0-training'),('receive','group-0')])
            self.assertEqual(packet['group'],item);self.assertTrue((output/'import-group-00000000.json').exists())


class RecoveryTests(unittest.TestCase):
    def fixture(self,root):
        root.mkdir();(root/'.run.lock').touch();(root/'policies').mkdir();(root/'native-serving/vllm-actor').mkdir(parents=True)
        artifact=root/'policies/payload.pt'
        payload={'version':0,'lineage':'same-original','identity':{'data_sha256':'b'*64},'sampling':{'temperature':.8,'top_p':1,'top_k':0},'trainable':{'backend.lm_head.weight':torch.zeros(2,3)},'frozen':{}}
        torch.save(payload,artifact);digest=hashlib.sha256(artifact.read_bytes()).hexdigest();path=artifact.with_name('policy-00000000-'+digest+'.pt');artifact.rename(path)
        value={k:payload[k] for k in ('version','lineage','identity','sampling')};value.update(path=str(path),digest=digest)
        policy=artifact_policy(value)
        (root/'cpu-policy-lease.json').write_text(json.dumps({'policy':policy.to_dict()}))
        (root/'mounted-vllm-qualification.json').write_text(json.dumps({'parameters_unchanged':True,'policy':policy.to_dict()}))
        shutdown=root/'native-serving/vllm-actor/actor-exit.json';shutdown.write_text(json.dumps({'returncode':0,'drain_complete':True,'actor_process':{'pid':99998,'start_ticks':2,'cmdline_sha256':'a'*64}}))
        owner=root.parent/'owner.json';owner.write_text(json.dumps({'pid':99999,'pid_start_ticks':1,'cmdline_sha256':'a'*64}))
        (root/'native-serving/owned-controller-startup.json').write_text(json.dumps({'model_pid':99999,'model_pid_start_ticks':1,'model_cmdline_sha256':'a'*64,'vllm_engine':{'actor_process':{'pid':99998,'start_ticks':2,'cmdline_sha256':'a'*64},'workers':[{'pid':99997,'start_ticks':3,'cmdline_sha256':'a'*64}]}}))
        stages=[{'stage':name,'status':'completed','pid':99999} for name in ['model_startup','initial_policy_publish','manager_startup','wait_eligible_groups','manager_close']]
        stages[-2].update(status='failed',error_type='ModuleNotFoundError',details={'step':0,'required_groups':1})
        (root/'learner-stage-timings.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in stages))
        return owner,digest,hashlib.sha256(shutdown.read_bytes()).hexdigest(),value,policy
    def test_original_policy_bytes_identity_and_lineage_remain_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()/'training';owner,digest,shutdown,value,policy=self.fixture(root)
            with patch('rollout_manager.terminal_bench.preupdate_recovery._exited') as exited, patch('rollout_manager.terminal_bench.preupdate_recovery.linux_process_identity',return_value={'model_pid':1}):
                actual,lease,proof,consumer=original_initial_policy(root,owner,expected_digest=digest,expected_actor_exit_sha256=shutdown,consumer_output=root.parent/'fresh',expected_group_seal_sha256='c'*64)
                consumer.close()
            self.assertEqual(actual,value);self.assertEqual(lease['policy'],policy.to_dict());self.assertEqual(exited.call_count,3)
            self.assertFalse(proof['optimizer_step_previously_applied']);self.assertFalse(proof['episode_generation_replayed'])
    def test_exclusive_claim_holds_kernel_lock_and_refuses_second_consumer(self):
        import fcntl
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()/'training';owner,digest,shutdown,_,_=self.fixture(root)
            with patch('rollout_manager.terminal_bench.preupdate_recovery._exited'), patch('rollout_manager.terminal_bench.preupdate_recovery.linux_process_identity',return_value={'model_pid':1}):
                _,_,_,consumer=original_initial_policy(root,owner,expected_digest=digest,expected_actor_exit_sha256=shutdown,consumer_output=root.parent/'fresh',expected_group_seal_sha256='c'*64)
                with (root/'.run.lock').open('r+') as another, self.assertRaises(BlockingIOError):fcntl.flock(another,fcntl.LOCK_EX|fcntl.LOCK_NB)
                consumer.close()
                with self.assertRaises(FileExistsError):
                    original_initial_policy(root,owner,expected_digest=digest,expected_actor_exit_sha256=shutdown,consumer_output=root.parent/'different',expected_group_seal_sha256='c'*64)
    def test_consumed_group_or_forced_shutdown_cannot_recover(self):
        for mode in ('optimizer','consumed','forced','artifact'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve()/'training';owner,digest,shutdown,_,_=self.fixture(root)
                if mode=='optimizer':
                    with (root/'learner-stage-timings.jsonl').open('a') as f:f.write(json.dumps({'stage':'train_groups','status':'failed','pid':99999})+'\n')
                elif mode=='consumed':(root/'import-group-00000000.json').write_text('{}')
                elif mode=='forced':
                    p=root/'native-serving/vllm-actor/actor-exit.json';x=json.loads(p.read_text());x['forced_termination']=True;p.write_text(json.dumps(x));shutdown=hashlib.sha256(p.read_bytes()).hexdigest()
                else:digest='f'*64
                with patch('rollout_manager.terminal_bench.preupdate_recovery._exited'),self.subTest(mode=mode),self.assertRaises(ValueError):
                    original_initial_policy(root,owner,expected_digest=digest,expected_actor_exit_sha256=shutdown,consumer_output=root.parent/'fresh',expected_group_seal_sha256='c'*64)
    def test_historical_delivery_binds_original_tokens_and_actual_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();output=root/'native-serving';output.mkdir();item=group();calls=[]
            class Delivery:
                def __init__(self,*args,**kwargs):pass
                def bind(self):calls.append('bind')
                def wait(self,ids):calls.append(ids);return {'delivery_drain_complete':True,'acknowledged_request_ids':ids}
            for sample in item.samples:
                t=sample.turns[0];p=output/(t.request_id+'.training.json');p.write_text(json.dumps(t.to_dict()));p.chmod(0o600)
                p=output/(t.request_id+'.http-state.json');p.write_text(json.dumps({'id':t.request_id,'http_response_written':True,'sampler_drained':True,'automatic_replay':False}));p.chmod(0o600)
            with patch('rollout_manager.terminal_bench.preupdate_recovery.LocalGPURelayDeliveryProbe',Delivery):
                validator=HistoricalGroupDelivery(root,root,18093,item.policy,'model','a'*64,'b'*64)
                validator._bind_original_relay=lambda:calls.append('bind')
                proof=validator.validate(item)
                self.assertFalse(proof['new_actor_claims_original_generation']);self.assertEqual(calls[-1],['0','1'])
                p=output/'0.training.json';x=json.loads(p.read_text());x['action_token_ids'][0]=4;p.write_text(json.dumps(x))
                with self.assertRaises(ValueError):validator.validate(item)


if __name__=='__main__':unittest.main()
