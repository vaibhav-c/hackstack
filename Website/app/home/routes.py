from app.home.plot import *
from app.home.dr import *
from app.home import blueprint
import time
from flask import render_template, redirect, url_for, request, session, jsonify
from jinja2 import TemplateNotFound
import csv
from datetime import date
import ast
import datetime
import json
import requests
import google.oauth2.credentials
import google_auth_oauthlib.flow
import googleapiclient.discovery
import numpy
import cv2
import base64
import io
from PIL import Image
import base64
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

def get_aggregate(fit_service, startTimeMillis, endTimeMillis, dataSourceId):
    return fit_service.users().dataset().aggregate(userId="me", body={
        "aggregateBy": [{
            "dataTypeName": "com.google.step_count.delta",
            "dataSourceId": dataSourceId
        }],
        "bucketByTime": {"durationMillis":86400000},
        "startTimeMillis": startTimeMillis,
        "endTimeMillis": endTimeMillis
    }).execute()

with open('app/base/static/assets/data/yoga_data.json') as json_file:
    yoga_data = json.load(json_file)

with open('app/base/static/assets/data/mental_data.json') as json_file:
    mental_data = json.load(json_file)

with open('app/base/static/assets/data/exercise_data.json') as json_file:
    exercise_data = json.load(json_file)

with open('app/base/static/assets/data/covid-data.json') as json_file:
    covid_data = json.load(json_file)

with open('app/base/static/assets/data/physical_pain.json') as json_file:
    physical_data = json.load(json_file)

with open('app/base/static/assets/data/nutrition_data.json') as json_file:
    nutrition_data = json.load(json_file)


@blueprint.route('/index',methods=["GET","POST"])

def index():
    
    events=[]
    try:
        credentials = google.oauth2.credentials.Credentials(**session['credentials'])
        if 'credentials' not in session:
            credentials = google.oauth2.credentials.Credentials(**session['credentials'])
        service = googleapiclient.discovery.build('calendar', 'v3', credentials=credentials)

        now = datetime.datetime.utcnow().isoformat() + 'Z' # 'Z' indicates UTC time
        #print('Getting the upcoming 10 events')
        events_result = service.events().list(calendarId='primary', timeMin=now,
                                            maxResults=10, singleEvents=True,
                                            orderBy='startTime').execute()
        events1 = events_result.get('items', [])
        
        calenderEvents=[]
        for event in events1:
            if 'dateTime' in event['start']:
                calenderEvents.append({"title":event["summary"],"start":event["start"]["dateTime"],"end":event["end"]["dateTime"]})
            elif "date" in event["start"]:
                calenderEvents.append({"title":event["summary"],"start":event["start"]["date"],"end":event["end"]["date"]})
    except:
        calenderEvents=[]
    with open('CSVs/Events.csv','r') as data:
        for line in csv.DictReader(data):
            line["Attending"]=ast.literal_eval(line["Attending"])
            events.append(line)
    events=sorted(events, key=lambda x: datetime.datetime.strptime(x["Start"], "%Y-%m-%d"))
    temp=events.copy()
    inEvent=[]
   
    for event in events:
        if(datetime.datetime.strptime(event ["Start"], "%Y-%m-%d") <datetime.datetime.today() ):
            temp.remove(event)
    events=temp.copy()


    return render_template('index.html', segment='index',events=events, cevents=calenderEvents,date=str(date.today()))

@blueprint.route('/<template>')

def route_template(template):

    try:

        if not template.endswith( '.html' ):
            template += '.html'

        # Detect the current page
        segment = get_segment( request )

        # Serve the file (if exists) from app/templates/FILE.html
        return render_template( template, segment=segment )

    except TemplateNotFound:
        return render_template('page-404.html'), 404
    
    except:
        return render_template('page-500.html'), 500



# employee entire work all graphs of indivituals to be shown here
@blueprint.route('/work')
def route_work_employee():
    

    return render_template('work_one.html', segment= get_segment(request))

# employee health
@blueprint.route('/health')
def route_health_individual():

    allDataSupplied = {
        'height_fit': 1.81,
        'weight_fit': 73,
        'steps_fit': 1000,
        'calorie_fit': 400,
        'nutrition_data': nutrition_data
    }

    try:
        credentials = google.oauth2.credentials.Credentials(**session['credentials'])
        if 'credentials' not in session:
            credentials = google.oauth2.credentials.Credentials(**session['credentials'])
        service = googleapiclient.discovery.build('fitness', 'v1', credentials=credentials)
        
        
        end_time_millis = int(round(time.time() * 1000))
        start_time_millis =  end_time_millis - 7 * 86400000
        steps = "derived:com.google.step_count.delta:com.google.android.gms:merge_step_deltas"
        calorie = "derived:com.google.calories.expended:com.google.android.gms:merge_calories_expended"
        height = "derived:com.google.height:com.google.android.gms:merge_height"
        #heart = "derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm"
        #sleep = "derived:com.google.sleep.segment:com.google.android.gms:sleep_from_activity<-raw:com.google.activity.segment:com.heytap.wearable.health:stream_sleep"
        weight = "derived:com.google.weight:com.google.android.gms:merge_weight"

        calory_data = get_aggregate(service, start_time_millis, end_time_millis, calorie)
        for daily_calory_data in calory_data['bucket']:
        # use local date as the key
            data_point = daily_calory_data['dataset'][0]['point']
            if data_point:
                calories = data_point[0]['value'][0]['fpVal']
                data_source_id = data_point[0]['originDataSourceId']
                daily_calories = {'calories': calories, 'originDataSourceId': data_source_id}
        print("safwrhgh",daily_calory_data)
        allDataSupplied['calorie_fit'] = daily_calories['calories']

        calory_data = get_aggregate(service, start_time_millis, end_time_millis, steps)
        print("asrdsafwrhgh",daily_calories['calories'])
        for daily_calory_data in calory_data['bucket']:
        # use local date as the key
            data_point = daily_calory_data['dataset'][0]['point']
            if data_point:
                calories = data_point[0]['value'][0]['intVal']
                data_source_id = data_point[0]['originDataSourceId']
                daily_calories = {'calories': calories, 'originDataSourceId': data_source_id}
        print("safwrhgh",daily_calory_data)
            
        allDataSupplied['steps_fit'] = daily_calories['calories']

        calory_data = get_aggregate(service, start_time_millis, end_time_millis, height)
        for daily_calory_data in calory_data['bucket']:
        # use local date as the key
            data_point = daily_calory_data['dataset'][0]['point']
            if data_point:
                calories = data_point[0]['value'][0]['fpVal']
                data_source_id = data_point[0]['originDataSourceId']
                daily_calories = {'calories': calories, 'originDataSourceId': data_source_id}
                allDataSupplied['height_fit'] = daily_calories['calories'] 
            else:
                allDataSupplied['height_fit']: 1.81

        calory_data = get_aggregate(service, start_time_millis, end_time_millis, weight)
        for daily_calory_data in calory_data['bucket']:
        # use local date as the key
            data_point = daily_calory_data['dataset'][0]['point']
            if data_point:
                calories = data_point[0]['value'][0]['fpVal']
                data_source_id = data_point[0]['originDataSourceId']
                daily_calories = {'calories': calories, 'originDataSourceId': data_source_id}
                allDataSupplied['weight_fit'] = daily_calories['calories']
            else :
                allDataSupplied['weight_fit']: 1.81     
    except:
        print("not signed in")


    return render_template('health.html', segment = get_segment(request),allData=allDataSupplied)


#employee basically skill groaph
@blueprint.route('/dental',methods=["GET","POST"])
def root():
    if request.method == 'POST':
        # save the single "profile" file
        if 'file' in request.files:
            img = cv2.imdecode(numpy.fromstring(request.files['file'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
            #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            nimage=cavity(img)
            #nimage=plaque(img)
            #nimage=staining(img)
            nimage = Image.fromarray(nimage.astype('uint8'))
            data = io.BytesIO()
            nimage.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())
            print(type(nimage))
            return render_template('dental.html', segment = get_segment(request),img=encoded_img_data.decode('utf-8'))
        elif 'file1' in request.files:
            img = cv2.imdecode(numpy.fromstring(request.files['file1'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
            #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            nimage=staining(img)
            #nimage=plaque(img)
            #nimage=staining(img)
            nimage = Image.fromarray(nimage.astype('uint8'))
            data = io.BytesIO()
            nimage.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())
            print(type(nimage))
            return render_template('dental.html', segment = get_segment(request),img1=encoded_img_data.decode('utf-8'))
        elif 'file2' in request.files:
            img = cv2.imdecode(numpy.fromstring(request.files['file2'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
            #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print("@@@@@@@@@@@@@@",type(img))
            nimage=plaque(img)
            #nimage=plaque(img)
            #nimage=staining(img)
            nimage = Image.fromarray(nimage.astype('uint8'))
            data = io.BytesIO()
            nimage.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())
            print(type(nimage))
            return render_template('dental.html', segment = get_segment(request),img2=encoded_img_data.decode('utf-8'))
        elif 'file3' in request.files:
            img = cv2.imdecode(numpy.fromstring(request.files['file3'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(type(img))
            dia=dr(img)
            return render_template('dental.html', segment = get_segment(request),dia=dia)
        elif 'file4' in request.files:
            img = cv2.imdecode(numpy.fromstring(request.files['file4'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(type(img))
            lung=lung1(img)
            return render_template('dental.html', segment = get_segment(request),lung=lung)


    return render_template('dental.html', segment = get_segment(request))

@blueprint.route('/plots/<template>',methods=["GET","POST"])
def oneskill(template):
    
    return render_template('one-skill.html', segment = get_segment(request))

@blueprint.route('/exercise')
def exercise():
    

    exercise_data['WLB'] = 4
    return render_template('exercise.html', segment = get_segment(request), allData = exercise_data)

@blueprint.route('/covid-faq')
def covidfaq():
    #print(covid_data)
    return render_template('covid-faq.html', segment = get_segment(request), allData = covid_data)

@blueprint.route('/physical-pain')
def physical():
    #print(physical_data)
    return render_template('physical-pain.html', segment = get_segment(request), allData = physical_data)

@blueprint.route('/yoga')
def yoga():
    return render_template('yoga.html', segment = get_segment(request), allData = yoga_data)

@blueprint.route('/yoga/prayanama/<template>')
def yoga_one(template):
    allDataSupplied = list(filter(lambda yo: yo['sanskrit_name'] == template, yoga_data['prayanama']))
    return render_template('yoga-one.html', segment = get_segment(request), allData = allDataSupplied[0])

@blueprint.route('/mental-health')
def mental():
    return render_template('mental-health.html', segment = get_segment(request), allData = mental_data)

@blueprint.route('/mental-health/<template>')
def mental_one(template):
    allDataSupplied = list(filter(lambda yo: yo['name'] == template, mental_data['conditions']))
    return render_template('mental-health-one.html', segment = get_segment(request), allData = allDataSupplied[0])



@blueprint.route('/company')
def route_rank():
    
    return render_template('rankings.html', segment = get_segment(request))

# Helper - Extract current page name from request 
def get_segment( request ): 

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment    

    except:
        return None  


# for admin


@blueprint.route('/skin',methods=["GET","POST"])
def skin():
    if request.method == 'POST':
        img = cv2.imdecode(numpy.fromstring(request.files['file'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(type(img))
        ans, prediction = skin1(img)
        return render_template('skin-disease.html', segment = get_segment(request), skin = ans, prediction = round(prediction * 100, 2))
    return render_template('skin-disease.html', segment = get_segment(request))


@blueprint.route('/lung',methods=["GET","POST"])
def lung():
    if request.method == 'POST':
        # save the single "profile" file
        img = cv2.imdecode(numpy.fromstring(request.files['file'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(type(img))
        lung, prediction = lung1(img)
        return render_template('lung-disease.html', segment = get_segment(request), lung = lung, prediction = round(prediction * 100, 2))
    return render_template('lung-disease.html', segment = get_segment(request))

@blueprint.route('/retina',methods=["GET","POST"])
def diabetes():
    if request.method == 'POST':
        img = cv2.imdecode(numpy.fromstring(request.files['file'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(type(img))
        diabetes, prediction = dr(img)
        return render_template('diabetes-er.html', segment = get_segment(request), diabetes = diabetes, prediction = round(prediction * 100, 2))
    return render_template('diabetes-er.html', segment = get_segment(request))

# GOOGLE


















# This variable specifies the name of a file that contains the OAuth 2.0
# information for this application, including its client_id and client_secret.
CLIENT_SECRETS_FILE = "client_secret.json"

# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account and requires requests to use an SSL connection.
SCOPES = ['https://www.googleapis.com/auth/fitness.body.read','https://www.googleapis.com/auth/fitness.activity.read','https://www.googleapis.com/auth/fitness.location.read','https://www.googleapis.com/auth/fitness.oxygen_saturation.read','https://www.googleapis.com/auth/fitness.reproductive_health.read','https://www.googleapis.com/auth/fitness.sleep.read','https://www.googleapis.com/auth/fitness.activity.read','https://www.googleapis.com/auth/calendar','https://www.googleapis.com/auth/calendar.events','https://www.googleapis.com/auth/calendar.events.readonly','https://www.googleapis.com/auth/calendar.readonly','https://www.googleapis.com/auth/calendar.settings.readonly']
API_SERVICE_NAME = 'Fitness API'
API_VERSION = 'v2'


@blueprint.route('/indextable')
def print_index_table1():
    return print_index_table()


@blueprint.route('/test')
def test_api_request():
    
    #     return redirect('authorize')

    # # Load credentials from the session.
    # credentials = google.oauth2.credentials.Credentials(
    #     **session['credentials'])

    # drive = googleapiclient.discovery.build(
    #     API_SERVICE_NAME, API_VERSION, credentials=credentials)

    # files = drive.files().list().execute()

    # # Save credentials back to session in case access token was refreshed.
    # # ACTION ITEM: In a production app, you likely want to save these
    # #              credentials in a persistent database instead.
    # session['credentials'] = credentials_to_dict(credentials)

    # return jsonify(**files)
    
    
    credentials = google.oauth2.credentials.Credentials(**session['credentials'])
    if 'credentials' not in session:
        credentials = google.oauth2.credentials.Credentials(**session['credentials'])
    service = googleapiclient.discovery.build('fitness', 'v1', credentials=credentials)
    
    daily_calories = {}
    end_time_millis = int(round(time.time() * 1000))
    start_time_millis =  end_time_millis - 100000000
    calory_data = get_aggregate(service, start_time_millis, end_time_millis, "derived:com.google.weight:com.google.android.gms:merge_weight")
    #return calory_data
    return calory_data
    for daily_calory_data in calory_data['bucket']:
        return daily_calory_data
        # use local date as the key
       
        data_point = daily_calory_data['dataset'][0]['point']
        if data_point:
            calories = data_point[0]['value'][0]['intVal']
            data_source_id = data_point[0]['originDataSourceId']
            daily_calories = {'calories': calories, 'originDataSourceId': data_source_id}

    return daily_calories

@blueprint.route('/authorize')
def authorize():
    # Create flow instance to manage the OAuth 2.0 Authorization Grant Flow steps.
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES)

    # The URI created here must exactly match one of the authorized redirect URIs
    # for the OAuth 2.0 client, which you configured in the API Console. If this
    # value doesn't match an authorized URI, you will get a 'redirect_uri_mismatch'
    # error.
    flow.redirect_uri = url_for('.oauth2callback', _external=True)
    #print(flow.redirect_uri)
    authorization_url, state = flow.authorization_url(
        # Enable offline access so that you can refresh an access token without
        # re-prompting the user for permission. Recommended for web server apps.
        access_type='offline',
        state = 'dummy',
        # Enable incremental authorization. Recommended as a best practice.
        include_granted_scopes='true')

    # Store the state so the callback can verify the auth server response.
    session['state'] = state
    #print(authorization_url)
    return redirect(authorization_url)


@blueprint.route('/oauth2callback')
def oauth2callback():
    # Specify the state when creating the flow in the callback so that it can
    # verified in the authorization server response.
    state = session['state']

    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES, state=state)
    flow.redirect_uri = url_for('.oauth2callback', _external=True)

    # Use the authorization server's response to fetch the OAuth 2.0 tokens.
    authorization_response = request.url
    flow.fetch_token(authorization_response=authorization_response)

    # Store credentials in the session.
    # ACTION ITEM: In a production app, you likely want to save these
    #              credentials in a persistent database instead.
    credentials = flow.credentials
    session['credentials'] = credentials_to_dict(credentials)

    return redirect('/index')


@blueprint.route('/revoke')
def revoke():
    if 'credentials' not in session:
        return ('You need to <a href="/authorize">authorize</a> before ' +
                'testing the code to revoke credentials.')

    credentials = google.oauth2.credentials.Credentials(
        **session['credentials'])

    revoke = requests.post('https://oauth2.googleapis.com/revoke',
        params={'token': credentials.token},
        headers = {'content-type': 'application/x-www-form-urlencoded'})

    status_code = getattr(revoke, 'status_code')
    if status_code == 200:
        return('Credentials successfully revoked.')
    else:
        return('An error occurred.')


@blueprint.route('/clear')
def clear_credentials():
    if 'credentials' in session:
        del session['credentials']
    return ('Credentials have been cleared.<br><br>')


def credentials_to_dict(credentials):
    return {'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes}

def print_index_table():
    return ('<table>' +
            '<tr><td><a href="/test">Test an API request</a></td>' +
            '<td>Submit an API request and see a formatted JSON response. ' +
            '    Go through the authorization flow if there are no stored ' +
            '    credentials for the user.</td></tr>' +
            '<tr><td><a href="/authorize">Test the auth flow directly</a></td>' +
            '<td>Go directly to the authorization flow. If there are stored ' +
            '    credentials, you still might not be prompted to reauthorize ' +
            '    the application.</td></tr>' +
            '<tr><td><a href="/revoke">Revoke current credentials</a></td>' +
            '<td>Revoke the access token associated with the current user ' +
            '    session. After revoking credentials, if you go to the test ' +
            '    page, you should see an <code>invalid_grant</code> error.' +
            '</td></tr>' +
            '<tr><td><a href="/clear">Clear Flask session credentials</a></td>' +
            '<td>Clear the access token currently stored in the user session. ' +
            '    After clearing the token, if you <a href="/test">test the ' +
            '    API request</a> again, you should go back to the auth flow.' +
            '</td></tr></table>')







