patient_lookup_outcome='''You are a medical admin assistant with one simple task, which is, given a text representation of a Epic EHR patient lookup window for some MRN (medical record number), classify whether:
		
		(1) The MRN matches exactly 1 patient
		(2) The MRN matches more than 1 patient
		(3) The MRN does not match any patients
		
		You simply have to respond either: '1', '2', or '3'

		Here are some examples:

		Example screen 1:
		========================
		<text id=0>Search for a Patient</text>
		<link id=1></link>
		<link id=2>L</link>
		<link id=3></link>
		<link id=4>Privacy View</link>
		<text id=5>Forename</text>
		<input id=6></input>
		<link id=7>MDT Conference</link>
		<text id=8>NHS No.</text>
		<link id=9></link>
		<input id=10>190</input>
		<link id=11></link>
		<link id=12></link>
		<button id=13>0 Find Patient</button>
		<link id=14></link>
		<text id=15>No patients were found .</text>
		<input id=16></input>
		<text id=17>Surname</text>
		<link id=18></link>
		<text id=19>Phone</text>
		<link id=20></link>
		<link id=21></link>
		<link id=22>Edit Assignments</link>
		<link id=23>➡Create New Patient</link>
		<link id=24>create a new patient</link>
		<text id=25>MRN</text>
		<link id=26>PROOF OF CONCEPT ( PO</link>
		<link id=27>Answer Pt - Qnr ( Captive )</link>
		<text id=28>Update the search terms to try again , or</text>
		<link id=29>Open Slots</link>
		<link id=30>tv</link>
		<text id=31>ID Type</text>
		<link id=32>U</link>
		<link id=33>Slicer Dicer</link>
		<link id=34>Personalise</link>
		<link id=35>My Incomplete Notes 1</link>
		<input id=36>O</input>
		<link id=37>Letter Queue 15</link>
		<link id=38>Procedure Catalogue</link>
		<text id=39>Patient</text>
		<link id=40>Sign My Visits</link>
		<input id=41>0</input>
		<link id=42>Devices</link>
		<link id=43>MDT</link>
		<link id=44>Status Board</link>
		<link id=45>NOA</link>
		<link id=46>16</link>
		<input id=47>□</input>
		<link id=48>Reschedule MDT Meeting</link>
		<link id=49>Schedule</link>
		<link id=50>Epic</link>
		<link id=51>Check In</link>
		<input id=52></input>
		<link id=53></link>
		<link id=54></link>
		<link id=55>Orders Only</link>
		<button id=56>Results</button>
		<link id=57>Tel Encounter</link>
		<link id=58>Telephone Call</link>
		<text id=59>Sex</text>
		<link id=60>www</link>
		<link id=61>/ Service Task</link>
		<link id=62></link>
		<link id=63>4</link>
		<link id=64>Notes</link>
		<link id=65></link>
		<link id=66>Remind Me</link>
		<link id=67></link>
		<link id=68>CT</link>
		<link id=69></link>
		<link id=70>View</link>
		<link id=71>Citrix Viewer</link>
		<text id=72>Postcode</text>
		<link id=73>Check Out</link>
		<link id=74>Abstract</link>
		<link id=75>JUL 25</link>
		<link id=76>3+</link>
		<input id=77></input>
		<link id=78>Go to In Basket for other tasks</link>
		<link id=79></link>
		<link id=80>☎</link>
		<link id=81>Print</link>
		<link id=82></link>
		<text id=83>Privacy View is on</text>
		<link id=84></link>
		<link id=85>Clear</link>
		<link id=86></link>
		<link id=87></link>
		<link id=88>Video Visit</link>
		<link id=89></link>
		<text id=90>Birthdate</text>
		<link id=91>GREAT ORMOND STREET ...</link>
		<input id=92></input>
		<link id=93>EpicCare</link>
		<link id=94>X</link>
		<link id=95>Hold MDT</link>
		<link id=96>42</link>
		<text id=97>Service Area</text>
		<link id=98>> Log Out</link>
		<text id=99>Start Time</text>
		<link id=100>G</link>
		<link id=101>Tue 25 Jul 11:03</link>
		<link id=102>Q</link>
		<link id=103></link>
		<link id=104>Status</link>
		<link id=105></link>
		<text id=106>TAN , CHRISTOPHER</text>
		<link id=107></link>
		<link id=108>X Sign Encounter</link>
		<link id=109>Patient Lookup</link>
		<link id=110>Chart</link>
		<link id=111>TAN , CHRISTOPHER</link>
		<link id=112>Print AVS</link>
		<link id=113></link>
		<link id=114>X Cancel</link>
		<text id=115>Perform a search on the local patient index</text>
		<link id=116>Patient Encounter</link>
		<link id=117></link>
		<link id=118></link>
		<text id=119>3 /</text>
		<link id=120>□</link>
		<link id=121>Patient Encounter</link>
		<text id=122>DNA</text>
		<text id=123>Patient Encounter</text>
		<link id=124></link>
		<link id=125></link>
		<link id=126></link>
		<link id=127></link>
		<link id=128></link>
		<link id=129></link>
		<text id=130>All Done !</text>
		<text id=131>My Patients</text>
		<text id=132>Recent Patients</text>
		<link id=133></link>
		<link id=134></link>
		<link id=135></link>
		<link id=136></link>
		<img id=137></img>
		<link id=138>X</link>
		<link id=139></link>
		<link id=140></link>
		<link id=141>L</link>
		<link id=142>www</link>
		<text id=143></text>
		<link id=144></link>
		<link id=145></link>
		<text id=146>Patient Lookup</text>
		<link id=147>My Patients</link>
		<link id=148></link>
		<link id=149></link>
		<link id=150>dh [ 6</link>
		<text id=151>190</text>
		<link id=152>tv</link>
		<link id=153>Privacy View is on .</link>
		<link id=154>?</link>
		<button id=155>X</button>
		<link id=156>www</link>
		<text id=157></text>
		<text id=158>?</text>
		<link id=159>Q</link>
		<link id=160></link>
		<text id=161></text>
		<link id=162></link>
		<text id=163></text>
		<input id=164>Accept</input>
		<link id=165></link>
		<link id=166></link>
		<input id=167>Recent Patients</input>
		<link id=168>DNA</link>
		<text id=169>TAN , CHRISTOPHER Privacy View is on .</text>
		<text id=170></text>
		<link id=171>Start Time</link>
		<link id=172></link>
		<img id=173></img>
		<img id=174></img>
		<text id=175>Hold MDT</text>
		<link id=176></link>
		<link id=177></link>
		<img id=178></img>
		<link id=179>1</link>
		<link id=180></link>
		<text id=181></text>
		<link id=182>42</link>
		<link id=183>/</link>
		<text id=184></text>
		<link id=185></link>
		<link id=186>Accept</link>
		<link id=187>All Done !</link>
		<text id=188>3</text>
		========================
		ANSWER: 3

		Example screen 2:
		========================
		<text id=0>Search for a Patient</text>
		<link id=1></link>
		<link id=2>L</link>
		<input id=3>3</input>
		<link id=4></link>
		<input id=5></input>
		<text id=6>Forename</text>
		<link id=7>Privacy View</link>
		<link id=8>MDT Conference</link>
		<link id=9></link>
		<input id=10>111</input>
		<link id=11></link>
		<link id=12></link>
		<link id=13></link>
		<link id=14></link>
		<link id=15></link>
		<link id=16></link>
		<text id=17>SESEMANN , Klara - 111</text>
		<text id=18>NHS No.</text>
		<text id=19>Surname</text>
		<link id=20>Edit Assignments</link>
		<text id=21>1619 South University</text>
		<text id=22>Ethnicity :</text>
		<text id=23>NHS Number :</text>
		<text id=24>No e - mail address on file</text>
		<link id=25>PROOF OF CONCEPT ( PO</link>
		<link id=26>Answer Pt - Qnr ( Captive )</link>
		<link id=27>➡Create New Patient</link>
		<text id=28>9 y.o. Female</text>
		<text id=29>608-251-7777 ( H</text>
		<link id=30>tv</link>
		<text id=31>Born 11/11/2013</text>
		<link id=32>Open Slots</link>
		<link id=33>U</link>
		<link id=34>Slicer Dicer</link>
		<link id=35>Personalise</link>
		<link id=36>My Incomplete Notes 1</link>
		<link id=37>Letter Queue 15</link>
		<input id=38>O</input>
		<button id=39>0 Find Patient</button>
		<link id=40>Procedure Catalogue</link>
		<text id=41>347 895 5467</text>
		<text id=42>Phone</text>
		<text id=43>MRN</text>
		<text id=44>England</text>
		<input id=45></input>
		<text id=46>BS6 7EY</text>
		<text id=47>: Decline to Answer</text>
		<link id=48>Sign My Visits</link>
		<link id=49>Devices</link>
		<link id=50>MDT Prep</link>
		<link id=51>Status Board</link>
		<text id=52>ID Type</text>
		<text id=53>Language :</text>
		<input id=54>0</input>
		<link id=55>16</link>
		<link id=56>Reschedule MDT Meeting</link>
		<link id=57>Epic</link>
		<link id=58>NOA</link>
		<link id=59>Check In</link>
		<link id=60>Schedule</link>
		<link id=61></link>
		<link id=62></link>
		<text id=63>Decline to Answer</text>
		<img id=64></img>
		<link id=65>Orders Only</link>
		<input id=66></input>
		<text id=67>Sex</text>
		<link id=68>Telephone Call</link>
		<text id=69>Patient</text>
		<link id=70>Tel Encounter</link>
		<link id=71>www</link>
		<text id=72>Race :</text>
		<text id=73>English</text>
		<input id=74>□</input>
		<link id=75>/ Service Task</link>
		<link id=76></link>
		<text id=77>Bristol</text>
		<link id=78>JUL 25</link>
		<link id=79>4</link>
		<link id=80></link>
		<link id=81>Remind Me</link>
		<link id=82></link>
		<link id=83>Notes</link>
		<link id=84>View</link>
		<link id=85>CT</link>
		<link id=86>Citrix Viewer</link>
		<link id=87>Check Out</link>
		<link id=88>This patient has the MRN that was entered</link>
		<link id=89></link>
		<link id=90>Go to In Basket for other tasks</link>
		<text id=91>Birthdate</text>
		<link id=92>Abstract</link>
		<link id=93>3+</link>
		<text id=94>11/11/2013</text>
		<link id=95>Date Of Birth</link>
		<link id=96>Secure</link>
		<link id=97></link>
		<link id=98>☎</link>
		<link id=99>Print Print</link>
		<link id=100></link>
		<button id=101>Results</button>
		<link id=102></link>
		<text id=103>Privacy View is on</text>
		<link id=104>GREAT ORMOND STREET ...</link>
		<text id=105>111</text>
		<link id=106>Video Visit</link>
		<link id=107></link>
		<text id=108>111</text>
		<text id=109>Service Area</text>
		<link id=110>✓Clear</link>
		<link id=111></link>
		<link id=112>EpicCare</link>
		<link id=113></link>
		<link id=114>Hold MDT</link>
		<link id=115>Status</link>
		<link id=116>42</link>
		<link id=117>Log Out</link>
		<text id=118>NHS No.</text>
		<link id=119>X Cancel</link>
		<text id=120>Nn GP on file</text>
		<text id=121>Start Time</text>
		<link id=122>G</link>
		<link id=123>Q</link>
		<text id=124>Match</text>
		<link id=125></link>
		<link id=126>Tue 25 Jul 11:06</link>
		<text id=127>TAN , CHRISTOPHER</text>
		<link id=128>Legal Sex</link>
		<link id=129></link>
		<link id=130></link>
		<text id=131>70.00</text>
		<text id=132>SESEMANN , KLARA</text>
		<link id=133></link>
		<input id=134></input>
		<link id=135>MRN</link>
		<text id=136>Postcode</text>
		<link id=137>Phone</link>
		<link id=138>Street Address</link>
		<link id=139>Patient Lookup</link>
		<link id=140>X Sign Encounter</link>
		<link id=141>Chart</link>
		<link id=142></link>
		<link id=143>Print AVS</link>
		<link id=144>Patient Encounter</link>
		<link id=145>TAN , CHRISTOPHER</link>
		<link id=146></link>
		<text id=147>3 /</text>
		<link id=148>Patients</link>
		<link id=149>11/11/2013</link>
		<link id=150>608-251-7777</link>
		<text id=151>Patient Name</text>
		<link id=152></link>
		<button id=153>✔Accept</button>
		<link id=154>Patient Encounter</link>
		<text id=155>DNA</text>
		<link id=156></link>
		<link id=157></link>
		<text id=158>Patient Encounter</text>
		<text id=159>608-251-7777</text>
		<link id=160></link>
		<link id=161></link>
		<text id=162>Perform a search on the local patient index</text>
		<link id=163></link>
		<link id=164></link>
		<link id=165></link>
		<input id=166>ID Type O</input>
		<text id=167>Bristol</text>
		<link id=168></link>
		<text id=169>All Done !</text>
		<link id=170></link>
		<link id=171></link>
		<img id=172></img>
		<link id=173>Legal Sex</link>
		<text id=174>MRN</text>
		<link id=175>X</link>
		<link id=176></link>
		<link id=177></link>
		<link id=178></link>
		<link id=179>L</link>
		<link id=180>www</link>
		<link id=181>✔Accept</link>
		<link id=182></link>
		<text id=183>Patient Lookup</text>
		<link id=184></link>
		<link id=185></link>
		<link id=186></link>
		<link id=187>Date Of Birth</link>
		<link id=188>tv</link>
		<link id=189>Results</link>
		<link id=190>Privacy View is on .</link>
		<link id=191>?</link>
		<button id=192>X</button>
		<link id=193>dh [ 6</link>
		<text id=194></text>
		<text id=195>Recent Patients</text>
		<link id=196>www</link>
		<text id=197></text>
		<text id=198>?</text>
		<link id=199>Q</link>
		<img id=200></img>
		<link id=201></link>
		<text id=202></text>
		<link id=203>Street Address</link>
		<link id=204></link>
		<link id=205>11/11/2013</link>
		<text id=206></text>
		<link id=207></link>
		<link id=208></link>
		<link id=209></link>
		<input id=210>Recent Patients</input>
		<link id=211>0 Find Patient</link>
		<text id=212></text>
		<input id=213>Street Address NHS No.</input>
		<link id=214>DNA</link>
		<text id=215>TAN , CHRISTOPHER Privacy View is on .</text>
		<link id=216></link>
		<link id=217></link>
		<link id=218>Start Time</link>
		<text id=219>Hold MDT</text>
		<img id=220></img>
		<img id=221></img>
		<link id=222></link>
		<link id=223>1</link>
		<img id=224></img>
		<link id=225>MRN</link>
		<text id=226>Language : English</text>
		<link id=227>All Done !</link>
		<text id=228></text>
		<link id=229></link>
		<link id=230>F</link>
		<link id=231>42</link>
		<link id=232>/</link>
		<link id=233></link>
		<text id=234></text>
		<link id=235></link>
		<input id=236>X Cancel</input>
		<button id=237></button>
		========================
		Answer: 1

		Example screen 3:
		========================
		<text id=0>Search for a Patient</text>
		<text id=1>01/01/2016</text>
		<text id=2>14/07/2015</text>
		<text id=3>England</text>
		<text id=4>England</text>
		<text id=5>01/01/2010</text>
		<text id=6>06/11/2010</text>
		<link id=7></link>
		<text id=8>England</text>
		<link id=9>L</link>
		<link id=10></link>
		<text id=11>Forename</text>
		<input id=12></input>
		<text id=13>2002323</text>
		<text id=14>10.57</text>
		<input id=15>20</input>
		<link id=16>Privacy View</link>
		<link id=17>MDT Conference</link>
		<link id=18></link>
		<link id=19></link>
		<link id=20></link>
		<text id=21>M</text>
		<link id=22></link>
		<input id=23></input>
		<link id=24></link>
		<link id=25></link>
		<text id=26>2002321</text>
		<link id=27></link>
		<text id=28>Surname</text>
		<link id=29>Edit Assignments</link>
		<text id=30>2002326</text>
		<text id=31>10.57</text>
		<text id=32>2002325</text>
		<text id=33>NHS No.</text>
		<text id=34>Legal Sex</text>
		<link id=35>PROOF OF CONCEPT ( PO</link>
		<link id=36>Answer Pt - Qnr ( Captive )</link>
		<text id=37>Phone</text>
		<link id=38>➡Create New Patient</link>
		<link id=39>tv</link>
		<link id=40>Open Slots</link>
		<input id=41>O</input>
		<link id=42>U</link>
		<link id=43>Slicer Dicer</link>
		<link id=44>Personalise</link>
		<link id=45>My Incomplete Notes 1</link>
		<link id=46>Letter Queue 15</link>
		<text id=47>MRN</text>
		<link id=48>Procedure Catalogue</link>
		<input id=49></input>
		<input id=50></input>
		<link id=51>Sign My Visits</link>
		<text id=52>ID Type</text>
		<link id=53>Devices</link>
		<link id=54>MDT</link>
		<text id=55>01234567890</text>
		<text id=56>11 Little Lane , Littlehampton , LH12 9UY , England</text>
		<link id=57>Status Board</link>
		<link id=58>16</link>
		<button id=59>0 Find Patient</button>
		<link id=60>Reschedule MDT Meeting</link>
		<input id=61>0</input>
		<link id=62>Epic</link>
		<link id=63>Check In</link>
		<link id=64></link>
		<link id=65>Schedule</link>
		<link id=66></link>
		<text id=67>Street Address</text>
		<link id=68></link>
		<text id=69>Sex</text>
		<text id=70>No phone numbers on file</text>
		<link id=71>Orders Only</link>
		<text id=72>M</text>
		<text id=73>10.57</text>
		<link id=74>ISLETS TRANSPLANT EVALUATION ACC ...</link>
		<link id=75>Telephone Call</link>
		<link id=76>www</link>
		<link id=77>Tel Encounter</link>
		<text id=78>NHS No.</text>
		<text id=79>Patient</text>
		<text id=80>MRN</text>
		<link id=81>KIDNEY TRANSPLANT EVALUATION ACC ...</link>
		<text id=82>Birthdate</text>
		<link id=83>Service Task</link>
		<input id=84>□</input>
		<text id=85>No e - mail address on file</text>
		<text id=86>Phone</text>
		<link id=87>JUL 25</link>
		<link id=88></link>
		<link id=89>4</link>
		<link id=90></link>
		<link id=91></link>
		<link id=92>Remind Me</link>
		<link id=93>CT</link>
		<link id=94>HEART TRANSPLANT EVALUATION ACCO</link>
		<link id=95>View</link>
		<link id=96>Notes</link>
		<link id=97></link>
		<link id=98>Citrix Viewer</link>
		<link id=99>DECEASED</link>
		<link id=100>Check Out</link>
		<link id=101>Abstract</link>
		<link id=102>3+</link>
		<img id=103></img>
		<text id=104>NHS Number : Not on file</text>
		<text id=105>No date of birth on file</text>
		<link id=106>Secure</link>
		<link id=107>Go to In Basket for other tasks</link>
		<text id=108>Privacy View is on</text>
		<text id=109>< E6340 ></text>
		<link id=110>INTESTINE TRANSPLANT EVALUATION A</link>
		<link id=111>☎</link>
		<text id=112>F</text>
		<link id=113>Print Print</link>
		<link id=114></link>
		<link id=115></link>
		<text id=116>Date of Birth</text>
		<link id=117></link>
		<link id=118></link>
		<link id=119>Video Visit</link>
		<img id=120></img>
		<text id=121>Service Area</text>
		<text id=122>No GP on file</text>
		<link id=123></link>
		<link id=124>EpicCare</link>
		<link id=125>NHS Number : Not on file</link>
		<link id=126></link>
		<text id=127>Start Time</text>
		<link id=128>✓Clear</link>
		<link id=129>Hold MDT</link>
		<text id=130>Recent Patients</text>
		<text id=131>F</text>
		<link id=132>42</link>
		<text id=133>CCB 20180812 TEST RSH , Research - < E6340 ></text>
		<link id=134>Log Out</link>
		<text id=135>20</text>
		<link id=136>X Cancel</link>
		<link id=137>G</link>
		<text id=138>TAN , CHRISTOPHER</text>
		<input id=139></input>
		<link id=140>Q</link>
		<link id=141>Tue 25 Jul 11:09</link>
		<link id=142></link>
		<link id=143></link>
		<link id=144>Status</link>
		<img id=145></img>
		<link id=146></link>
		<text id=147>Bridge Four , Shattered Plains , ZZ99 3WZ , Engl</text>
		<link id=148>Patient Lookup</link>
		<button id=149>Results</button>
		<text id=150>Postcode</text>
		<link id=151>X Sign Encounter</link>
		<text id=152>CCB 20180812 TEST RSH , RESEARCH ( ak ...</text>
		<link id=153>Patient Encounter</link>
		<text id=154>Aliases : 20180812</text>
		<link id=155>Chart</link>
		<link id=156></link>
		<link id=157>Results</link>
		<link id=158>Print AVS</link>
		<link id=159></link>
		<text id=160>Perform a search on the local patient index</text>
		<text id=161>3 /</text>
		<text id=162>My Patients</text>
		<text id=163>Legal Sex : Not on file</text>
		<text id=164>10.57</text>
		<text id=165>Gender Identity : Not on file</text>
		<text id=166>DNA</text>
		<link id=167></link>
		<link id=168></link>
		<link id=169></link>
		<text id=170>England</text>
		<link id=171></link>
		<text id=172>Patient Encounter</text>
		<input id=173>ID Type O</input>
		<link id=174></link>
		<link id=175></link>
		<link id=176>Patient Name</link>
		<link id=177></link>
		<img id=178></img>
		<link id=179></link>
		<link id=180></link>
		<link id=181>✓ Accept</link>
		<link id=182></link>
		<link id=183></link>
		<link id=184></link>
		<img id=185></img>
		<link id=186></link>
		<button id=187>✓ Accept</button>
		<link id=188></link>
		<link id=189>X</link>
		<text id=190>All Done !</text>
		<link id=191></link>
		<link id=192>L</link>
		<link id=193>www</link>
		<link id=194></link>
		<link id=195></link>
		<text id=196>Patient Lookup</text>
		<text id=197>Patient Name</text>
		<link id=198></link>
		<link id=199>NA</link>
		<link id=200>tv</link>
		<img id=201></img>
		<link id=202>?</link>
		<link id=203>dh [ 6</link>
		<button id=204>X</button>
		<link id=205>Date of Birth</link>
		<link id=206>www</link>
		<link id=207>GREAT ORMOND STREET ...</link>
		<text id=208></text>
		<text id=209>Match</text>
		<text id=210>?</text>
		<link id=211>Q</link>
		<link id=212>Privacy View is on .</link>
		<link id=213></link>
		<text id=214></text>
		<link id=215></link>
		<text id=216>GREAT ORMOND STREET ...</text>
		<text id=217></text>
		<link id=218></link>
		<text id=219>TAN , CHRISTOPHER Privacy View is on .</text>
		<link id=220>Gender Identity : Not on file</link>
		<link id=221></link>
		<text id=222></text>
		<text id=223>10.57</text>
		<link id=224>DNA</link>
		<text id=225>Bridge Four</text>
		<link id=226>0 Find Patient</link>
		<link id=227>Patient Encounter</link>
		<text id=228>E6340 ></text>
		<link id=229></link>
		<link id=230>All Done !</link>
		<text id=231>F</text>
		<link id=232></link>
		<text id=233>Hold MDT</text>
		<img id=234></img>
		<img id=235></img>
		<text id=236>11 Little Lane ,</text>
		<text id=237>10.57</text>
		<link id=238>1</link>
		<link id=239></link>
		<img id=240></img>
		<img id=241></img>
		<link id=242></link>
		<text id=243></text>
		<link id=244>42</link>
		<link id=245>TAN , CHRISTOPHER</link>
		<link id=246>/</link>
		<text id=247></text>
		<link id=248></link>
		<link id=249>Start Time</link>
		<link id=250>Patient Encounter</link>
		<link id=251></link>
		<link id=252></link>
		========================
		Answer: 2

		The current screen now follows:
        '''