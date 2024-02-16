import os
from vllm import SamplingParams
from vllm import LLM
from huggingface_hub import snapshot_download


class InferlessPythonModel:
    def initialize(self):
        self.template = """SYSTEM: You are a helpful assistant.
        USER: {}
        ASSISTANT: """
        local_path = "/var/nfs-mount/translation-pipeline-volume"
        if os.path.exists(local_path) == False :
            os.makedirs(local_path)
            snapshot_download(
                "meta-llama/Llama-2-7b-chat-hf",
                local_dir=local_path,
                token="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",
            )
    
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
        result_output = {"result" : "works"}

        return {"result": result_output[0]}

    def finalize(self, args):
        pass
