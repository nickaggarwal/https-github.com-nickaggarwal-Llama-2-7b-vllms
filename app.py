import os
from vllm import SamplingParams
from vllm import LLM
from huggingface_hub import snapshot_download


class InferlessPythonModel:
    def initialize(self):
        self.template = """SYSTEM: You are a helpful assistant.
        USER: {}
        ASSISTANT: """
        local_path = "/var/nfs-mount/translation-pipeline-volume/model4"
        if os.path.exists(local_path) == False :
            os.makedirs(local_path)
            snapshot_download(
                "meta-llama/Llama-2-7b-chat-hf",
                local_dir=local_path,
                local_dir_use_symlinks=False ,
                token="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"
            )
        self.llm = LLM(
          model="/var/nfs-mount/translation-pipeline-volume/model4",
          dtype="float16")
    
    def infer(self, inputs):
        print("inputs[questions] -->", inputs["questions"], flush=True)
        prompts = [self.template.format(inputs["questions"])]
        print("Prompts -->", prompts, flush=True)
        sampling_params = SamplingParams(
            temperature=0.75,
            top_p=1,
            max_tokens=250,
            presence_penalty=1.15,
        )
        return {"result" : "works"}

    def finalize(self, args):
        pass
