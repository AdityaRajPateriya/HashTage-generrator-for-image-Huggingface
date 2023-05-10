import streamlit as st
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer
import openai
from itertools import cycle
from tqdm import tqdm
from PIL import Image
import torch
import os

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

openai.api_key = "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"
openai_model = "text-davinci-002"


def hashtag_generator(des):
    hashtag_prompt = ('''Please generation ten relative and accurate hashtag that will help the photo reach a large 
    audience on Instagram and Twitter for a photo that shows ''' + des + '''.The hashtag can be funny and 
    creative.Please also provide in this format. Hashtags: #[Hashtag1]
    #[Hashtag2]  #[Hashtag3]
    #[Hashtag4]  #[Hashtag5]
    #[Hashtag7]  #[Hashtag6] 
    #[Hashtag8]  #[Hashtag7]
    #[Hashtag9]
    #[Hashtag10]''')

    # Hashtag Generation
    response = openai.Completion.create(
        engine=openai_model,
        prompt=hashtag_prompt,
        max_tokens=(20 * 10),
        n=1,
        stop=None,
        temperature=0.7,
    )
    hashtag = response.choices[0].text.strip().split("\n")
    return hashtag


def prediction(img_list):
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    img = []

    for image in tqdm(img_list):
        i_image = Image.open(image)
        st.image(i_image, width=200)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode='RGB')

        img.append(i_image)

    pixel_val = processor(images=img, return_tensors="pt").pixel_values
    pixel_val = pixel_val.to(device)

    output = model.generate(pixel_val, **gen_kwargs)

    predict = tokenizer.batch_decode(output, skip_special_tokens=True)
    predict = [pred.strip() for pred in predict]

    return predict



def upload():
    # from uploader inside tab
    with st.form("uploader"):
        # Image input
        image = st.file_uploader("upload Images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])
        # generate button
        submit = st.form_submit_button("Generate")
        if submit:
            description = prediction(image)
            st.subheader("#Hashtags")
            hashtags = hashtag_generator(description[0])
            for hash in hashtags:
                st.write(hash)


def main():
    st.set_page_config(page_title="hashtag generator")
    st.title("Cool HeshTag Generater for Images")
    st.subheader('by Aditya raj Pateriya')

    tab1 = st.tabs("Upload your images")
    upload()


if __name__ == '__main__':
    main()
