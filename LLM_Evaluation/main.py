import os
from langchain.chat_models import init_chat_model
from langsmith import Client

API_KEY = ""
# Configure environment variables for LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = API_KEY

client = Client(api_key=API_KEY)

# Initialize Ollama model with LangChain (temperature=0 for deterministic evaluation)
llm = init_chat_model("deepseek-r1:1.5b", model_provider="ollama", temperature=0)


# datasets

dataset_name = "Test Dataset v2"
if not client.has_dataset(dataset_name=dataset_name):
    data_sets = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        dataset_id=data_sets.id,
        examples=[
            {
                "inputs": {
                    "question": "What is the capital of France?",
                },
                "outputs": {
                    "answer": "Paris",
                },
            },
            {
                "inputs": {
                    "question": "What is the langchain",
                },
                "outputs": {
                    "answer": "Langchain is a framework for developing applications that use language models.",
                }
            },
        ],
    )

# LLM AS A JUDGE
eval_instructions = "You are an expert professor specialized in grading students' answers to questions."

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    user_content = f"""You are grading the following question:
    {inputs.get('question', '')}
    Here is the real answer:
    {reference_outputs.get('answer', '')}
    You are grading the following predicted answer:
    {outputs.get('response', '')}
    Respond with CORRECT or INCORRECT:
    Grade:
    """
    response = llm.invoke(
        [
            ("system", eval_instructions),
            ("user", user_content)
        ]
    ).content

    # DeepSeek-R1 outputs thought blocks (<thought>...</thought>) which we need to strip
    if "</thought>" in response:
        response = response.split("</thought>")[-1]
    response_clean = response.strip().upper()

    if "INCORRECT" in response_clean:
        return False
    return "CORRECT" in response_clean

def concision(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    pred = outputs.get("response") or ""
    ref = reference_outputs.get("answer") or ""
    return len(pred) < 2 * len(ref)


default_instructions = "Respond to the users question in a short, concise manner (one short sentence)."
def my_app(question: str, model: str = "gpt-4o-mini", instructions: str = default_instructions) -> str:
    response = llm.invoke(
        [
            ("system", instructions),
            ("user", question),
        ],
    ).content

    # Strip thought blocks from the DeepSeek-R1 response for concise answer checking
    if "</thought>" in response:
        response = response.split("</thought>")[-1]
    return response.strip()

    
### Call my_app for every datapoint
def ls_target(inputs: dict) -> dict:
    return {"response": my_app(inputs.get("question", ""))}



## Run our evaluation
experiment_results = client.evaluate(
    ls_target,  # Your AI system
    data=dataset_name,
    evaluators=[correctness, concision],
    experiment_prefix="openai-4o-mini-chatbot"
)
