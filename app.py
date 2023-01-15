import streamlit as st

def run_app():
    st.title("My Streamlit App")

    if st.checkbox('Show Hello World'):
        st.write('Hello, World!')

if __name__ == "__main__":
    run_app()


