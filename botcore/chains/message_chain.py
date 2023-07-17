from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain import LLMChain

def build_message_chain(model):
    # define the output
    response_schemas = [
     ResponseSchema(name="sentiment", description="a sentiment label based on the past content. It should be either Negative or Positive"),
     ResponseSchema(name="ideas", description="a js array of response options to the given content. These options must help the user to feel better."),
     ResponseSchema(name="chain", description="always return 'message'")
     ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    # prompt template
    template = """You are good at detecting harmful, content and also encouraging good and useful one. All labels that you know are Negative and Positive. Given a past content, your job is to answer as best as possible.
The past message:
{input}

Instructions:
{format_instructions}
The question: What do you think about the message."""

    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(template=template, input_variables=["input"],
                        partial_variables={"format_instructions": format_instructions},
                        output_parser=output_parser)

    # Build chain
    chain = LLMChain(llm=model, prompt=prompt, verbose=True)
    return chain
