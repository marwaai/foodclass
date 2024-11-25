 echo "BUILD START"
 python3.9  pip install -r requirements.txt
 python3.9  pip install psycopg2-binary==2.8.4
 python3.9 manage.py collectstatic --noinput --clear
 echo "BUILD END"
