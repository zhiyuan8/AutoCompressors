import torch
from transformers import AutoTokenizer
from auto_compressor import LlamaAutoCompressorModel, AutoCompressorModel
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="`do_sample` is set to `False`. However, `temperature` is set to")
warnings.filterwarnings("ignore", message="`do_sample` is set to `False`. However, `top_p` is set to")

# Load AutoCompressor trained by compressing 6k tokens in 4 compression steps
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/AutoCompressor-Llama-2-7b-6k")
# Need bfloat16 + cuda to run Llama model with flash attention
model = LlamaAutoCompressorModel.from_pretrained("princeton-nlp/AutoCompressor-Llama-2-7b-6k", torch_dtype=torch.bfloat16).eval().cuda()

prompt = "Explain the significance of Red Hat's acquisition of NooBaa."
prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda()

context = """With Red Hat, IBM to become the leading hybrid cloud provider Watch Now After IBM acquired Red Hat, I suggested IBM paid $34 billion for the Linux power so it could become a hybrid-cloud power. With the news that Red Hat will acquire NooBaa, a hybrid-cloud, data-storage company, it's become clearer than ever that the IBM-Red Hat deal is all about the hybrid cloud. NooBaa is a software-defined storage company. Its primary program of the same name is open-source software, which puts a virtual layer over private and public clouds storage resources. Also: IBM: Chip making is hitting its limits, but our techniques could solve that It's made up of three components: First, there's an access node, which handles the data chunking, deduplication, compression and encryption between storage resources; next, there's a storage daemon, which presents server storage as storage nodes; and finally, there's a virtual machine (VM) based core for data placement, self-healing, and monitoring. The Access nodes and storage daemons make up a data plane, while the core provides its control plane. Also: How IBM Watson is revolutionizing 10 industries TechRepublic So, what does all mean for customers? It's multi-cloud storage management, which enables allows you to manage, deploy, and migrate data storage across private and major public clouds. This includes Alibaba, AWS, Azure, and Google Cloud. It's easy to see why Red Hat values this. It gives their customers a way to manage storage without sweating the details across multiple platforms. As Ranga Rangachari, Red Hat's vice president of Storage and Hyperconverged Infrastructure, said in a statement: "Data portability is a key imperative for organizations building and deploying cloud-native applications across private and multiple clouds. NooBaa's technologies will augment our portfolio and strengthen our ability to meet the needs of developers in today's hybrid and multicloud world. We are thrilled to welcome a technical team of nine to the Red Hat family as we work together to further solidify Red Hat as a leading provider of open hybrid-cloud technologies." Related stories:"""
context_tokens = tokenizer(context, add_special_tokens=False, return_tensors="pt").input_ids.cuda()

summary_vectors = model(context_tokens, output_softprompt=True).softprompt
print(f"Compressing {context_tokens.size(1)} tokens to {summary_vectors.size(1)} summary vectors")

generation_with_summary_vecs = model.generate(prompt_tokens, do_sample=False, softprompt=summary_vectors, max_new_tokens=512)[0]
print("=== Generation w/ summary vectors ===\n" + tokenizer.decode(generation_with_summary_vecs))

next_tokens_without_context = model.generate(prompt_tokens, do_sample=False, max_new_tokens=512)[0]
print("=== Generation w/o context ===\n" + tokenizer.decode(next_tokens_without_context))