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

    tab1, tab2, tab3 = st.tabs(["**Home**", "**Generate Captions**", "**Train Model**"])

    with tab1:
        home_header = """<p style="font-size: 20px; text-align:justify;">
        The Images Caption Generator app allows users to upload the desired images to be captioned and get the suitable caption for each image. 
        Results are shown on the app page, and the user also has the choice to download the captioning results as a CSV file.
        <br><br>
        <strong>Train Model tab: </strong>Train a new neural network model using configurable parameters. <em>only available if the app is running locally</em>
        <br>
        <strong>Generate Captions tab: </strong>Upload images to be captioned and get the results.
        <br><br>
        For more information and instructions, see <a href="https://github.com/ZaidGhazal/Image-Captions-Generator">GitHub Repo</a>
        </p>
        """
        
        st.markdown(home_header, unsafe_allow_html=True)
        
        st.write("----------")
        st.markdown("""<p style="font-size: 33px; font-weight: bold;">
        How are captions being generated?
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col2.image("./assets/architecture_img.png", use_column_width=True,
        caption="The Network Workflow: input image, encoder, decoder, and resulted caption")

        home_body = """<p style="font-size: 20px; text-align:justify;">Inspired by <a href="https://arxiv.org/pdf/1411.4555.pdf">Show and Tell: A Neural Image Caption Generator</a> paper, a neural network was built and trained to extract features from the images and generate text sequence.
        The network architecture consists of two parts: The Encoder and Decoder.
        The Encoder consists of the pretrained ResNet-50 model layers (except the last fully connected linear layer) and an embedding linear layer used to get the extracted features vector and produce the embedding vector in a configurable size.
        ResNet-50 was chosen due to its power in classifying objects, which makes it excellent in extracting complex features through its convolutional network. 
        <br><br>
        The Decoder has three main parts: an embedding layer, an LSTM layer, and a fully connected output layer.
        The embedding layer is used to convert the input word indices to dense vectors of fixed size. The LSTM layer is a type of recurrent neural network that is used for processing sequential data. It has a configurable number of hidden states and a configurable number of layers. The fully connected output layer used to generate the final caption.
        <br><br>
        The deployed model has trained on the <a href="https://cocodataset.org/#home">COCO-2017</a> dataset. The training process was done on a machine equipped with a GeForce RTX 2080 Ti GPU. Training time was around 3 hours and 30 minutes. Training configurable parameters were as follows:
        <br><br>
        <ul>
        <li><strong>Batch Size:</strong> 128</li>
        <li><strong>Vocabulary Threshold:</strong> 8</li>
        <li><strong>Embedding Size:</strong> 400</li>
        <li><strong>Hidden States Size:</strong> 356</li>
        <li><strong>Learning Rate:</strong> 0.001</li>
        <li><strong>Number of Epochs:</strong> 2</li>
        </ul>
        </p>"""

        st.markdown(home_body, unsafe_allow_html=True)
        st.write("----------")
        
    with tab2:
        st.subheader("Upload Images to Generate Captions")
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
                # with st.expander("**See Images/Captions**"):
                    for image, caption in imgs_captions_list:
                        
                        st.image(image, use_column_width=True)
                        st.markdown("<p style='text-align: center; '>{}</p>".format(caption.strip()[:-1].strip().capitalize()), unsafe_allow_html=True)
                        st.write("---------------")
    
    with tab3:
        st.subheader("Train New Model with Specified Parameters")
        st.write("")
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
                    

                




if __name__ == "__main__":
    run_app()


