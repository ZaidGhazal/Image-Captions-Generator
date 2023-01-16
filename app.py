import streamlit as st

def run_app():
    st.title("Images Caption Generator")

    tab1, tab2, tab3 = st.tabs(["Home", "Train", "Inference"])

    with tab1:
        st.write("Home")
    
    with tab2:
        st.write("Here you can train a new neural netowrk model. You should specifiy values to the following paramters and hit the trian button to start.")
        
    
    with tab3:
        st.write("Here you can generate a caption for an image. You should select an image and hit the generate button to start.")
        

if __name__ == "__main__":
    run_app()


