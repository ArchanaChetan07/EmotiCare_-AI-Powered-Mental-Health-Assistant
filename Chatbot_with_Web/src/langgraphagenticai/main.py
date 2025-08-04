import streamlit as st
from src.langgraphagenticai.ui.streamlitui.loadui import LoadStreamlitUI
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.ui.streamlitui.display_result import DisplayResultStreamlit

def load_langgraph_agenticai_app():
    """
    Loads and runs the LangGraph AgenticAI application with Streamlit UI.
    Initializes the UI, handles user input, configures the LLM,
    sets up the graph for the selected use case, and displays output.
    """

    # Step 1: Load UI controls (Sidebar values)
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("❌ Error: Failed to load user input from the UI.")
        return

    # Step 2: Get user message from chat input
    user_message = st.chat_input("💬 Enter your message:")

    if user_message:
        try:
            # Step 3: Load the LLM using Groq settings
            obj_llm_config = GroqLLM(user_controls_input=user_input)
            model = obj_llm_config.get_llm_model()

            if not model:
                st.error("❌ Error: LLM model could not be initialized.")
                return

            # Step 4: Get selected use case (e.g., "Chatbot With Web")
            usecase = user_input.get("selected_usecase")
            if not usecase:
                st.error("❌ Error: No use case selected.")
                return

            # Step 5: Build LangGraph with selected model + use case
            graph_builder = GraphBuilder(model)

            try:
                graph = graph_builder.setup_graph(usecase)
                print("📨 User message:", user_message)

                # Step 6: Display result
                DisplayResultStreamlit(usecase, graph, user_message).display_result_on_ui()

            except Exception as e:
                st.error(f"❌ Error during graph setup: {e}")
                return

        except Exception as e:
            st.error(f"❌ LLM setup or graph invocation failed: {e}")
            return
