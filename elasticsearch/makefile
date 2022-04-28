es-start:
	docker run --name es01 -p 9200:9200 -p 9300:9300 -it docker.elastic.co/elasticsearch/elasticsearch:8.1.3
	docker cp es01:/usr/share/elasticsearch/config/certs/http_ca.crt .

clean:
	docker container rm es01