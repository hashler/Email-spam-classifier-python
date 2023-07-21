from flask import Flask, jsonify, render_template, request
import requests
import json, sys

from email_spam_classifier import getEmailScamPrediction

application = Flask(__name__)

@application.route('/')
def index():

    return render_template("index.html")

@application.post('/mail')
def mailPredict():
        mailtext = request.form['mail']

        ans = getEmailScamPrediction(mailtext)

        return render_template("mailPrediction.html", mailPrediction = ans, mailName = mailtext)

if __name__ == '__main__':
    application.run(debug=True)