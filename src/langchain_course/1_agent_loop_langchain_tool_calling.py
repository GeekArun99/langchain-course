from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

#--- tool (Langchain @tool decorator) ----

@tool 
def get_product_price(product : str) ->float:
    """Look up for the price of a product in the catalog."""
    print(f"  >> Executing get_product_price(product'{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard":89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier : str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers : broze, silver, gold."""
    print(f" >> Executing apply_discount(price='{price}', discount_tire='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver":12, "gold":23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


#-------------Agent loop-----------
@traceable(name = "LangChain Agent Loop")
def run_agent(question : str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name : t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [
        SystemMessage(
            content =(
                "You are a helpful shopping assistnant"
                "You have the access to the product catalog tool"
                "and a discount tool.\n\n"
                "STRICT RULES - You musy folow exactly:\n"
                "1.Never guess or assume any product price."
                "You MUST call get_product_price first to get the real price \n"
                "2. On;y call apply_discount AFTER you have recieved "
                "a price from get_product_price. Pass the exact price"
                "returned by get_product_price - don NOT pass a made_up number.\n"
                "3.NEVER calculate discounts yourself by using MATH."
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier"
                "ask them which tier to use -Do Not assume one"
            )
        ),
        HumanMessage(content=question) 
    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n---Iteration {iteration}---")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer :{ai_message.content}")
            return ai_message.content


        # Execute tools
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(
                f">> Calling {tool_name} "
                f"with {tool_args}"
            )

            tool_result = tools_dict[tool_name].invoke(tool_args)

            print(f">> Tool result: {tool_result}")

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
            )
if __name__=="__main__":
    print(f"Hello Langchain Agent (.bind tools)!")
    print()
    result = run_agent("What is the price of the laptop after applying gold discount?")

