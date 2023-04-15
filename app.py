#importing modules
import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st 
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
     
    return angle 
st.sidebar.title('Exercise Monitoring System')
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Knee Bend Exercise'])

if app_mode == 'Home':
    st.title('About Our Project')
    st.markdown("This app monitors the rep count of left knee bend exercise.")
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQbaayyXtu706xcwHXiP2l5TFYFDzcv6Ep3sg&usqp=CAU',caption='Exercise Monitoring System',use_column_width=True,)

elif app_mode == 'Knee Bend Exercise':
    header = st.container()
    
    FRAME_WINDOW = st.image([])
    with header:
        st.title('Exercise Monitoring System')
        st.text("This app monitors the rep count of left knee bend exercise.")
        st.text("Exercise description -")
        st.text("●Leg should be bent to start timer")
        st.text("●Slight inward bend is enough to start the timer. ( <140 deg)")
        st.text("●After a successful rep, the person has to stretch his/her leg straight.")
        st.text("●No restriction for back angle.")
        st.text("●Consider leg closer to camera as exercised leg.")
        st.text("●If you fails to stay in holding position till 8 sec.Feedback message is displayed")
        st.text(".     'Keep your knee bent for atleast 8 seconds.' ")

    run = st.checkbox('Ready')
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('https://drive.google.com/file/d/1QM8UFQciXxJll5SZaTAhdqO4mg6tzM8r/view?usp=sharing')
    # Set video camera size
    cap.set(3,1280)
    cap.set(4,960)
    
    # Bent counter variables
    counter = 0 
    stage = None
    
    # Time counter and message
    msg = None
    end=[]
    dtime=0

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        if run :
            while cap.isOpened():
                ret, frame = cap.read()

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                try:
                
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    leftknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y] #shoulder
                    lefthip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]  #elbow
                    leftankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y] #wrist

                    # Calculate angle
                    angle = calculate_angle(lefthip,leftknee,leftankle)
                    # Visualize angle
                    cv2.putText(image, str(round(angle,2)), 
                                   tuple(np.multiply(leftknee, [640, 480]).astype(int)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    if angle>140 and stage=='Left Knee Bend':
                        if dtime>=8:
                            counter+=1
                    # Curl counter logic
                    if angle > 140:
                        stage = "Straight"
                        msg='Your knee is straight.'

                    if angle <=140 and stage =='Straight':
                        stage="Left Knee Bend"
                        end.clear()

                    if angle<=140:
                        s=round(time.time(),2)
                        end.append(s)
                        dtime=end[-1]-end[0]
                        dtime=round(dtime,2)

                    if 1<=dtime<8:
                        msg='Keep your knee bent for atleast 8 seconds.'

                    elif dtime>=8:
                        msg='Well Done! Your knee bent have crossed 8 seconds.'

                    if angle > 140:
                        stage = "Straight"
                        msg='Your knee is straight.'

                except:
                    pass
                
                # Render bent counter
                # Setup status box

                cv2.rectangle(image, (0,0), (1000,90), (245,117,16), -1)
                cv2.putText(image,msg, (30,75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                # Rep data
                cv2.putText(image, 'REPS:', (15,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (110,35), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2, cv2.LINE_AA)

                # Stage data
                cv2.putText(image, 'STAGE:', (160,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, (265,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

                #time
                cv2.putText(image, 'Bent time(s):', (535,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image,str(dtime), (750,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))               

                cv2.imshow('Mediapipe Feed', image)
                FRAME_WINDOW.image(image)
                #Press esc key to end.
                if cv2.waitKey(10) & 0xFF == 27:
                    break
                
            cap.release()
            cv2.destroyAllWindows()
