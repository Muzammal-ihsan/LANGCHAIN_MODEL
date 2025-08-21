from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=""  # 👈 Add here
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("WHO IS IMRAN KHAN?")
print(result.content)
