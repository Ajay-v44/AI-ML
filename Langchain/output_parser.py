from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser,CommaSeparatedListOutputParser

model=ChatOllama(model="deepseek-r1:1.5b",temperature=0,top_p=0)
str_op_parser=StrOutputParser()
response=model.invoke("Hello, how are you?")
print(str_op_parser.invoke(response))

comma_separated_list_output_parser=CommaSeparatedListOutputParser()
response=model.invoke("list 10 fruits")
print(comma_separated_list_output_parser.invoke(response))


message_h=HumanMessage(content=f'''list 10 fruits
        {comma_separated_list_output_parser.get_format_instructions()}
        ''')

response=model.invoke([message_h])
print(response.content)
print(comma_separated_list_output_parser.invoke(response))

# DateTimeParser 
from datetime import datetime
from pydantic import BaseModel, Field

# 1. Define your schema
class DateResponse(BaseModel):
    date: datetime = Field(description="The current date and time parsed into a native object.")

# 2. Bind the schema directly to your model (No parsers or format instructions needed)
structured_model = model.with_structured_output(DateResponse)

# 3. Invoke the model directly
parsed_data = structured_model.invoke("what is today date?")

# This immediately yields a fully validated Pydantic object
print(parsed_data.date)
print(type(parsed_data.date))

