import asyncio
import copy
from dataclasses import replace
import json
import tempfile
import threading
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.rl.algorithm import response_log_probs
from dsv41_train.rl.checkpoint import publish_policy_artifact, run_identity
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.family import build_policy, load_model_config
from rollout_manager.backends.pytorch_controller import PyTorchPolicyController, SHELL_PARAMETERS, shell_completion
from rollout_manager.generation_training import TrainingGenerationContract
from rollout_manager import model_completion
from qwen_helpers import tiny_config, write_hf_checkpoint


class PyTorchControllerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        write_hf_checkpoint(self.root, Qwen3MoeForCausalLM(tiny_config(max_position_embeddings=128)))
        vocabulary = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, **{f"token{i}": i for i in range(3, 32)}}
        native = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        native.pre_tokenizer = pre_tokenizers.Whitespace()
        native.add_special_tokens(["[UNK]", "[BOS]", "[EOS]"])
        native.save(str(self.root / "tokenizer.json"))
        (self.root / "tokenizer_config.json").write_text('{"tokenizer_class":"PreTrainedTokenizerFast","unk_token":"[UNK]","bos_token":"[BOS]","eos_token":"[EOS]"}')
        (self.root / "chat_template.jinja").write_text(
            "{% for m in messages %}{{ m.role }} {{ m.content | default('', true) }} {% endfor %}assistant")
        (self.root / "data.jsonl").write_text('{"prompt":"fixture"}\n')
        self.config = RLConfig(rollout_engine="native-pytorch", max_new_tokens=3, temperature=.8, trainable_scope="lm_head")
        self.learner = build_policy(self.root, load_model_config(self.root), random_init=False,
            devices=("cpu",), dtype=torch.float32, trainable_scope="lm_head")
        self.identity = run_identity(self.root, self.root / "data.jsonl", False)
        self.identity["trainable_scope"] = "lm_head"
        self.sampling = {"temperature": .8, "top_p": 1, "top_k": 0}
        self.controller = PyTorchPolicyController(self.root, self.config, served_model="fixture/qwen")
        self.addAsyncCleanup(self.controller.close)
        self.payload = {"model": "fixture/qwen", "messages": [{"role": "user", "content": "token3 token4"}],
            "tools": [{"type": "function", "function": {"name": "shell", "description": "Run shell.",
                        "parameters": SHELL_PARAMETERS}}], "tool_choice": "auto", "parallel_tool_calls": False,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": False, "temperature": .8, "max_tokens": 3}

    def artifact(self, version):
        return publish_policy_artifact(self.learner, version, self.root / "policies", lineage="controller-test",
                                       identity=self.identity, sampling=self.sampling)

    async def start(self):
        artifact = self.artifact(0)
        self.assertEqual(await self.controller.start(artifact), {key: artifact[key] for key in ("version", "digest", "lineage")})
        return artifact

    async def test_real_pretrained_sampling_binding_replay_and_new_artifact(self):
        initial = await self.start()
        payload_before = copy.deepcopy(self.payload)
        completion = await asyncio.to_thread(self.controller.generate, self.payload, "a" * 32)
        contract = TrainingGenerationContract(self.controller.policy, self.controller.capabilities())
        trace = contract.validate(self.payload, completion, "a" * 32, 1)
        probabilities = response_log_probs(self.learner, trace.context_token_ids, trace.action_token_ids, .8)
        torch.testing.assert_close(probabilities, torch.tensor(trace.behavior_log_probs), rtol=0, atol=0)
        self.assertEqual(self.payload, payload_before)
        self.assertEqual(trace.policy, self.controller.policy_factory(initial))
        with self.assertRaisesRegex(ValueError, "no replay"):
            self.controller.generate(self.payload, "a" * 32)
        tampered = copy.deepcopy(completion)
        tampered["choices"][0]["message"]["content"] += " altered"
        with self.assertRaisesRegex(ValueError, "bind the raw completion"):
            contract.validate(self.payload, tampered, "a" * 32, 1)
        with torch.no_grad():
            self.learner.backend.lm_head.weight.add_(.05)
        newer = self.artifact(1)
        with self.assertRaisesRegex(RuntimeError, "proven paused"):
            await self.controller.activate(newer)
        pause = await self.controller.pause()
        self.assertTrue(pause["drain_complete"])
        with self.assertRaisesRegex(RuntimeError, "paused"):
            self.controller.generate(self.payload, "b" * 32)
        self.assertEqual(await self.controller.activate(newer), {key: newer[key] for key in ("version", "digest", "lineage")})
        torch.testing.assert_close(self.controller.model.backend.lm_head.weight,
                                   self.learner.backend.lm_head.weight, rtol=0, atol=0)
        await self.controller.resume()
        updated = self.controller.generate(self.payload, "b" * 32)
        self.assertEqual(updated["training_evidence"]["policy"]["version"], 1)

    async def test_raw_eos_is_retained_with_its_actual_probability(self):
        prompt = self.controller.tokenizer.encode_conversation(self.payload["messages"], tools=self.payload["tools"],
                                                             chat_template_kwargs={"enable_thinking": False})
        tokens = torch.tensor([prompt])
        with torch.no_grad():
            hidden = self.learner.backend.model(tokens, torch.ones_like(tokens)).last_hidden_state[0, -1]
            self.learner.backend.lm_head.weight.zero_()
            self.learner.backend.lm_head.weight[2].copy_(hidden * 100)
        await self.start()
        completion = self.controller.generate(self.payload, "c" * 32)
        trace = TrainingGenerationContract(self.controller.policy, self.controller.capabilities()).validate(
            self.payload, completion, "c" * 32, 1)
        self.assertEqual(trace.action_token_ids, (2,))
        self.assertEqual(trace.stop_token_count, 1)
        self.assertEqual(trace.finish_reason, "eos")
        self.assertEqual(completion["choices"][0]["message"]["content"], "")
        self.assertEqual(trace.behavior_log_probs, (0.,))

    async def test_cancelled_caller_does_not_fake_sampler_drain(self):
        await self.start()
        newer = self.artifact(1)
        entered, release = threading.Event(), threading.Event()
        forward = self.controller.model.forward
        def blocking(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test sampler release missing")
            return forward(*args, **kwargs)
        self.controller.model.forward = blocking
        sampling = asyncio.create_task(asyncio.to_thread(self.controller.generate, self.payload, "d" * 32))
        self.assertTrue(await asyncio.to_thread(entered.wait, 5))
        sampling.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await sampling
        pausing = asyncio.create_task(self.controller.pause())
        await asyncio.sleep(.03)
        self.assertFalse(pausing.done())
        activating = asyncio.create_task(self.controller.activate(newer))
        await asyncio.sleep(.03)
        self.assertFalse(activating.done())
        self.assertEqual(self.controller.policy.version, 0)
        release.set()
        self.assertTrue((await pausing)["drain_complete"])
        self.assertEqual((await activating)["version"], 1)
        self.assertEqual(self.controller._active, 0)

    async def test_bad_payload_and_failed_load_never_serve(self):
        await self.start()
        for change in [{"temperature": .9}, {"top_p": .9}, {"stream": True}, {"parallel_tool_calls": True},
                       {"max_tokens": 4}, {"seed": 1}, {"chat_template_kwargs": {"enable_thinking": True}},
                       {"model": "other"}, {"tools": []}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.controller.generate({**self.payload, **change}, "e" * 32)
        self.assertEqual(self.controller._seen, set())
        await self.controller.pause()
        bad = {**self.artifact(1), "digest": "f" * 64}
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            await self.controller.activate(bad)
        with self.assertRaises(RuntimeError):
            await self.controller.resume()
        with self.assertRaises(RuntimeError):
            self.controller.generate(self.payload, "e" * 32)

    async def test_last_position_profile_is_bound_before_artifact_loading(self):
        from dsv41_train.rl.qwen_policy import LAST_POSITION_LOGITS_PATH

        config = replace(self.config, qwen_last_position_only=True, backprop_chunk_tokens=2)
        unqualified = PyTorchPolicyController(self.root, config, served_model="fixture/qwen")
        self.addAsyncCleanup(unqualified.close)
        with self.assertRaisesRegex(ValueError, "selected Qwen logits profile"):
            await unqualified.start(self.artifact(0))
        self.controller = PyTorchPolicyController(self.root, config, served_model="fixture/qwen")
        self.addAsyncCleanup(self.controller.close)
        self.identity["qwen_logits_path"] = LAST_POSITION_LOGITS_PATH
        self.learner.qwen_last_position_only = True
        await self.start()
        completion = self.controller.generate(self.payload, "f" * 32)
        trace = TrainingGenerationContract(self.controller.policy, self.controller.capabilities()).validate(
            self.payload, completion, "f" * 32, 1)
        self.assertEqual(trace.policy.identities["qwen_logits_path"], LAST_POSITION_LOGITS_PATH)
        torch.testing.assert_close(response_log_probs(self.learner, trace.context_token_ids, trace.action_token_ids, .8),
                                   torch.tensor(trace.behavior_log_probs), rtol=0, atol=0)

    async def test_cached_profile_is_bound_replayed_and_reset_after_activation(self):
        from dsv41_train.rl.qwen_policy import LAST_POSITION_LOGITS_PATH
        from rollout_manager.backends.qwen_kv_cache import QWEN_KV_SAMPLING_PATH

        config = replace(self.config, qwen_last_position_only=True, qwen_kv_cache=True)
        self.controller = PyTorchPolicyController(self.root, config, served_model="fixture/qwen")
        self.addAsyncCleanup(self.controller.close)
        self.identity["qwen_logits_path"] = LAST_POSITION_LOGITS_PATH
        with self.assertRaisesRegex(ValueError, "selected Qwen sampling profile"):
            await self.controller.start(self.artifact(0))
        self.controller = PyTorchPolicyController(self.root, config, served_model="fixture/qwen")
        self.addAsyncCleanup(self.controller.close)
        self.identity["qwen_sampling_path"] = QWEN_KV_SAMPLING_PATH
        self.learner.qwen_last_position_only = self.learner.qwen_kv_cache = True
        await self.start()
        for version in (0, 1):
            if version:
                await self.controller.pause()
                with torch.no_grad():
                    self.learner.backend.lm_head.weight.mul_(.9)
                await self.controller.activate(self.artifact(version))
                await self.controller.resume()
            completion = self.controller.generate(self.payload, str(version) * 32)
            trace = TrainingGenerationContract(self.controller.policy, self.controller.capabilities()).validate(
                self.payload, completion, str(version) * 32, 1)
            self.assertEqual(trace.policy.identities["qwen_sampling_path"], QWEN_KV_SAMPLING_PATH)
            self.assertEqual(trace.policy.version, version)
            torch.testing.assert_close(response_log_probs(self.learner, list(trace.context_token_ids),
                                                          list(trace.action_token_ids), .8),
                                       torch.tensor(trace.behavior_log_probs), rtol=0, atol=0)


class QwenShellParsingTests(unittest.TestCase):
    def test_original_shell_envelope_and_truncation(self):
        self.assertIs(shell_completion, model_completion.shell_completion)
        self.assertIs(SHELL_PARAMETERS, model_completion.SHELL_PARAMETERS)
        text = 'Inspecting.\n<tool_call> {"name":"shell","arguments":{"cmd":"printf \'你好\\n\'"}} </tool_call>'
        message, finish = shell_completion(text, "f" * 32)
        self.assertEqual(finish, "tool_calls")
        self.assertEqual(message["content"], "Inspecting.")
        self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]), {"cmd": "printf '你好\n'"})
        truncated, finish = shell_completion(text + "<tool_call>", "f" * 32, truncated=True)
        self.assertEqual((truncated, finish), ({"role": "assistant", "content": text + "<tool_call>"}, "length"))
        for invalid in ["<tool_call>{}</tool_call>", "<tool_call>{}", "</tool_call>",
                        '<tool_call>{"name":"other","arguments":{"cmd":"pwd"}}</tool_call>',
                        '<tool_call>{"name":"shell","name":"shell","arguments":{"cmd":"pwd"}}</tool_call>',
                        '<tool_call>{"name":"shell","arguments":{"cmd":"pwd","extra":1}}</tool_call>',
                        '<tool_call>{"name":"shell","arguments":{"cmd":NaN}}</tool_call>', "<think>hidden</think>"]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                shell_completion(invalid, "f" * 32)

    def test_more_than_four_calls_have_stable_distinct_ids_and_all_commands_are_checked(self):
        commands = ["pwd", "printf READY", "ls", "true", "git status", "printf DONE"]
        text = "\n".join('<tool_call>' + json.dumps({"name": "shell", "arguments": {"cmd": command}})
                         + '</tool_call>' for command in commands)
        message, finish = shell_completion(text, "a" * 32)
        self.assertEqual(finish, "tool_calls")
        self.assertEqual(message, shell_completion(text, "a" * 32)[0])
        calls = message["tool_calls"]
        self.assertEqual([call["id"] for call in calls], ["call_" + "a" * 32 + "_" + str(i) for i in range(6)])
        self.assertEqual([json.loads(call["function"]["arguments"])["cmd"] for call in calls],
                         commands)
        for bad in ("", "x" * 8193, "nul\0command"):
            invalid = text + '<tool_call>' + json.dumps(
                {"name": "shell", "arguments": {"cmd": bad}}) + '</tool_call>'
            with self.subTest(bad=bad[:20]), self.assertRaises(ValueError):
                shell_completion(invalid, "a" * 32)


if __name__ == "__main__":
    unittest.main()
