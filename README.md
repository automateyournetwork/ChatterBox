# ChatterBox
Talk to your Netbox with a Langchain 

## Getting started

Clone the repo

Make a .env file inside the packet_buddy folder (where the chatterbox.py file is located; /chatterbox/chatterbox/)

put this in the file:
```console
OPENAI_API_KEY="<your openapi api key>"
NETBOX_TOKEN='<your netbox API token>'
NETBOX_URL='https://demo.netbox.dev' (replace with the URL of your NetBox) 
```

## Bring up the server
docker-compose up 

## Visit localhost
http://localhost:8585

