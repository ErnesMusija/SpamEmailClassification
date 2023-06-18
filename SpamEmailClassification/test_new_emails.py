from joblib import load
from prepared_data import X_train, all_emails, real_emails
import pandas as pd

model = load('spam_email_rf_model.joblib')
vectorizer = load('vectorizer.joblib')

sender = '"Martin Adamson" <martin@srv0.ems.ed.ac.uk>'
recipient = 'jzzzzteana@yahoogroups.com'
subject = '[zzzzteana] Playboy wants to go out with a bang'
date = 'Thu, 22 Aug 2002 14:54:25 +0100'
content = 'The Scotsman - 22 August 2002 Playboy wants to go out with a bang    AN AGEING Berlin playboy has come up with an unusual offer to lure women into his bed - by promising the last woman he sleeps with an inheritance of 250,000 (£160,000).   Rolf Eden, 72, a Berlin disco owner famous for his countless sex partners, said he could imagine no better way to die than in the arms of an attractive young woman - preferably under 30.   "I put it all in my last will and testament - the last woman who sleeps with me gets all the money," Mr Eden told Bild newspaper.   "I want to pass away in the most beautiful moment of my life. First a lot of fun with a beautiful woman, then wild sex, a final orgasm - and it will all end with a heart attack and then Im gone."   Mr Eden, who is selling his nightclub this year, said applications should be sent in quickly because of his age. "It could end very soon," he said.------------------------ Yahoo! Groups Sponsor ---------------------~-->4 DVDs Free +s&p Join Nowhttp://us.click.yahoo.com/pt6YBB/NXiEAA/mG3HAA/7gSolB/TM---------------------------------------------------------------------~->To unsubscribe from this group, send an email to:forteana-unsubscribe@egroups.com Your use of Yahoo! Groups is subject to http://docs.yahoo.com/info/terms/'

email = pd.DataFrame({
    'Sender': [sender],
    'Recipient': [recipient],
    'Subject': [subject],
    'Date': [date],
    'Content': [content],
})
mails = real_emails[['Content', 'Subject', 'Date', 'Recipient', 'Sender']]
test = mails.head(10)
test = vectorizer.transform(test)
email = vectorizer.transform(email)

print(model.predict(email))

print(real_emails.iloc[7])
