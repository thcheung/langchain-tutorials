import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

messages = [
    {"role": "system", "content": "Translate the following from English into Traditional Chinese"},
    {"role": "user", "content": "hi!"}
    ]

for token in model.stream(messages):
    print(token.content)

from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Traditional Chinese", "text": "hi!"})

for token in model.stream(prompt):
    print(token.content)