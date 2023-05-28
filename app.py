from flask import Flask, render_template, request
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image

app = Flask(__name__)

dic = {0 :'BrownSpot', 1 : 'Healthy',2:'Hispa', 3:'LeafBlast'}

#modelresnet50dict
model = torch.load('Entire_resnet50_model.pth', map_location=torch.device('cpu'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict(path):
    data_transforms = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #file = request.files['file']
    # Preprocess the image using the data_transforms pipeline
    img = Image.open(path)
    img = data_transforms(img)
    img = img.unsqueeze(0)  # Add batch dimension
    # Make a prediction using the pre-trained model
    with torch.no_grad():
        output = model(img)
        # Convert the output to a numpy array
        output_np = output.numpy()
        # Convert the output to a list of class probabilities
        probs = np.exp(output_np) / np.sum(np.exp(output_np), axis=1)
        classes = ['BrownSpot ट्राइकोडर्मा (Tricyclazole): यह एक प्रमुख कीटनाशक है जो ब्राउन स्पॉट के खिलाफ प्रभावी होता है। इसे उच्च दर्जे में रोग प्रबंधन के लिए उपयोग किया जाता है। बेंकोजेब (Benzobisazole): यह एक अन्य प्रमुख कीटनाशक है जो ब्राउन स्पॉट रोग के खिलाफ प्रभावी होता है। इसे अनुशंसित खेतीकरों द्वारा उपयोग किया जाता है। कार्बेंडाजिम (Carbendazim): यह भी ब्राउन स्पॉट चावल रोग के खिलाफ लाभकारी है और रोग प्रबंधन में उपयोग किया जाता है। मेटालैक्सील (Metalaxyl): यह भी ब्राउन स्पॉट के खिलाफ प्रभावी होता है और कुछ उत्पादों में उपयोग किया जाता है। ',
                   'Healthy',
                   'Hispa "हिस्पा रोग को रोकने के लिए, किसान भाई इन दो स्प्रे का उपयोग कर सकते हैं: क्लोरान्ट्रानिलिप्रोल 18.5% SC @ 150ml/ha (35 और 75 दिन प्रतिस्थापन के बाद) ने हिस्पा, वर्ल मैगट और काला खटमल की सबसे अधिक कमी को कम कर दिया है (नियंत्रण के मुकाबले 91.80, 92.25, 84.51 प्रतिशत कमी), जिसे क्लोथियानिडिन 50 WDG @ 40g/ha (88.46, 89.60 और 83.39 प्रतिशत कमी) ने अनुसरण किया',
                   'LeafBlast ट्राइसाइकलाजोल: ट्राइसाइकलाजोल एक उच्चतम प्रभावशील सिस्टेमिक फंगाइसाइड है जो धान के रोगों, समेत लीफ ब्लास्ट के प्रबंधन के लिए विशेष रूप से विकसित किया गया है। यह फंगस की विकास रोककर और बीजांकन को रोककर उत्कृष्ट नियंत्रण प्रदान करता है। ट्राइसाइकलाजोल धान की लीफ ब्लास्ट प्रबंधन के लिए व्यापक रूप से प्रयोग किया जाता है और सिफारिश किया जाता है।']
        class_probs = {class_name: prob.item() for class_name, prob in zip(classes, probs[0])}
        #print(class_probs)
    #img.close() # Close the file
    # Return the predicted class as a JSON response
    return class_probs

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("home.html")

@app.route("/index.html", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "We are from GCET, and you are using our web app for predicting crop disease!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict(img_path)
        predicted_class = max(p, key=p.get)
        #print(predicted_class)

    return render_template("index.html", prediction = predicted_class, img_path = img_path)

if __name__ =='__main__':
    app.run(debug = True)
