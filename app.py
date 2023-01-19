import time
import pandas as pd
import streamlit as st
from inference import run_inference
from train import run_train
from multiprocessing import Pipe, Pool, Process, cpu_count
import os
import signal

st.set_page_config(layout="wide")

@st.experimental_memo
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

def start_train_process(
    batch_size, vocab_threshold, embed_size, hidden_size, learning_rate, num_epochs
):
    train_process = Process(target=run_train, args=(batch_size, vocab_threshold, embed_size, hidden_size, learning_rate, num_epochs))
    train_process.start()
    st.session_state['train_process'] = train_process
    st.session_state['train_pid'] = train_process.pid
    st.session_state['train_terminate'] = False
    st.info("Training Started!")

def run_app():

    st.title("Images Caption Generator")

    tab1, tab2, tab3 = st.tabs(["**Home**", "**Train**", "**Inference**"])

    with tab1:
        st.write("Home")
    
    with tab2:
        st.subheader("Here you can train a new neural netowrk model. You should specifiy values to the following paramters and hit the trian button to start.")
        col1, col2 = st.columns(2)
        with col1:
            learning_rate  = st.selectbox(
                        '**Select the Learning Rate**',
                        (0.0001, 0.001, 0.01))
            st.write("Learning Rate: ", learning_rate)
            
            st.write("")
            st.write("-------------")

            batch_size = st.selectbox(
                        '**Select Batch Size**',
                        (32, 64, 128, 256, 512))
            st.write('Batch Size:', batch_size)
            
            st.write("-------------")
            
            hidden_size = st.selectbox(
                        '**Select the RNN Hidden State Output Size**',
                        (64, 128, 256, 512))
            st.write('LSTM Hidden State Output Size:', hidden_size)

            
        
        with col2:
            embed_size = st.slider(
                        '**Select the Embedding Size**',
                        100, 2500, 300)
            st.write('Embedding Size:', embed_size)

            st.write("-------------")
            
            vocab_threshold = st.number_input(
                '**Select a Vocab Threshold**',
                3, 15, 8)
            st.write('Vocab Threshold:', vocab_threshold)

            st.write("-----------")
            num_epochs = st.number_input(
                    '**Select Epochs Number**',
                    1, 20, 1)
            st.write('Training Epochs:', num_epochs)

        st.write("-----------")
        
        
        if st.button("Train Network Model"):
            if st.session_state.get('train_process') is not None:
                if st.session_state.get('train_process').is_alive():
                        st.warning("Training is in progress!")   
                else:
                    start_train_process(batch_size, vocab_threshold, embed_size, hidden_size, learning_rate, num_epochs)
                    
            else:
                start_train_process(batch_size, vocab_threshold, embed_size, hidden_size, learning_rate, num_epochs)
        st.write("-------------")

        if st.session_state.get('train_process') is not None:
            col1, col2, col3, col4, _ = st.columns(5)

            with col2:
                if st.button("Check Status"): 
                    if st.session_state.get('train_process') is None:
                        st.warning("No Training is in progress!")
                    elif st.session_state.get('train_terminate'):
                        st.error("Training was Terminated. Please Train again.")
                    
                    elif not st.session_state.get("train_process").is_alive():
                            st.success("Training is Done!")
                    else:
                        st.info("Training is in progress...")      

            with col4:
                if st.session_state.get('train_process').is_alive():
                    if st.button("Stop Training"):
                        if st.session_state.get("train_pid") is None:
                            st.warning("No Training is in progress!")
                        else:
                            os.kill(st.session_state.get("train_pid"), signal.SIGTERM)
                            st.warning("Training is Stopped!")
                            print("Training was STOPED!", "Process Killed: ", st.session_state['train_pid'])
                            st.session_state['train_terminate'] = True

    with tab3:
        st.subheader("Here you can generate a caption for an image. You should select an image and hit the generate button to start.")
        # read img in streamlit
        imgs_path = st.file_uploader("Upload an image", type=["jpg", "png", "svg"], accept_multiple_files=True)
        if imgs_path != []:
            with st.spinner('Wait for it...'):
                imgs_captions_list = run_inference(imgs_path)
                
                # Get list of images names
                imgs_names = [path.name for path in imgs_path]
                # Get list of catptions
                captions = [caption.strip()[:-1].strip() for _, caption in imgs_captions_list]
                results_df = pd.DataFrame({"Image": imgs_names, "Caption": captions})
                csv =  convert_df(results_df)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name='images_captions.csv',
                    mime='text/csv'
                )


                col1, col2, col3 = st.columns(3)
                with col2:
                    with st.expander("**See Images/Captions**"):
                        for image, caption in imgs_captions_list:
                            st.image(image, use_column_width=True)
                            st.markdown("<p style='text-align: center; color: black;'>{}</p>".format(caption.strip()[:-1].strip().capitalize()), unsafe_allow_html=True)
                            st.write("---------------")
                    

                




if __name__ == "__main__":
    run_app()


