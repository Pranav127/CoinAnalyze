from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 2.5)  # Wait time between tasks for each simulated user

    @task(3)  # Frequency weight for this task
    def index(self):
        self.client.get("/")

    @task(2)
    def predict_get(self):
        self.client.get("/predict")

    @task(2)
    def predict_post(self):
        # This is a POST request. Provide necessary data for your route.
        self.client.post("/predict", data={"coin": "BTC", "model_type": "RF"})

    @task(2)
    def tradesignal(self):
        self.client.get("/tradesignal")

    @task(2)
    def get_signal(self):
        self.client.post("/get_signal", data={"symbol": "BTC-USD"})

    @task(2)
    def get_data(self):
        self.client.post("/get_data", data={"symbol": "BTC-USD", "page": "1"})

    @task(2)
    def get_chart(self):
        self.client.post("/get_chart", data={"symbol": "BTC-USD", "indicator": "all"})

    @task(1)
    def error_500(self):
        self.client.get("/500")

    @task(2)
    def news(self):
        # Assuming this is a GET request.
        self.client.get("/news")

    # Add tasks for any other routes you have in your Flask application

