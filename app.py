import numpy as np
from flask import Flask,jsonify,render_template,request
import pickle


# from keras import models
file=open('model.pkl','rb')
clf=pickle.load(file)

#file.close()

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def hello_world():
    if request.method == 'POST':

        erythema = int(request.form['erythema'])
        scaling = int(request.form['scaling'])
        definite = int(request.form['definite'])
        itching = int(request.form['itching'])
        koebner = int(request.form['koebner'])
        polygonal = int(request.form['polygonal'])
        follicular = int(request.form['follicular'])
        oral = int(request.form['oral'])
        knee = int(request.form['knee'])
        scalp = int(request.form['scalp'])
        family = int(request.form['family'])
        melanin = int(request.form['melanin'])
        eosinophils = int(request.form['eosinophils'])
        PNL = int(request.form['PNL'])
        fibrosis = int(request.form['fibrosis'])
        exocytosis = int(request.form['exocytosis'])
        acanthosis = int(request.form['acanthosis'])
        hyperkeratosis = int(request.form['hyperkeratosis'])
        parakeratosis = int(request.form['parakeratosis'])
        clubbing = int(request.form['clubbing'])
        elongation = int(request.form['elongation'])
        thinning = int(request.form['thinning'])
        spongiform = int(request.form['spongiform'])
        munro = int(request.form['munro'])
        focal = int(request.form['focal'])
        disappearance = int(request.form['disappearance'])
        vacuolisation = int(request.form['vacuolisation'])
        spongiosis = int(request.form['spongiosis'])
        saw_tooth = int(request.form['saw-tooth'])
        follicular = int(request.form['follicular'])
        perifollicular = int(request.form['perifollicular'])
        inflammatory = int(request.form['inflammatory'])
        band = int(request.form['band'])
        Age = int(request.form['Age'])
        
        
        input_feature = [erythema,scaling,definite,itching,koebner,polygonal,follicular,oral,knee,scalp,family,melanin,eosinophils,PNL,fibrosis,exocytosis,acanthosis,hyperkeratosis,parakeratosis,clubbing,elongation,thinning,spongiform,munro,focal,disappearance,vacuolisation,spongiosis,saw_tooth,follicular,perifollicular,inflammatory,band,Age]
        #input_feature=[5,1,1,1,2,1,3,1,1]
        infprob=clf.predict([input_feature])
        
        #infprob = round(infprob*100,4)
        return render_template('result.html',inf=infprob)
   
    return render_template('home.html')
   
if __name__ == '__main__'  :
    app.run(debug=False) 
