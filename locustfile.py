from locust import HttpUser, task

class FaceUser(HttpUser):

    @task
    def recognize(self):
        with open("sample.jpg", "rb") as f:
            self.client.post("/recognize", files={"file": f})