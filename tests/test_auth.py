def login(client, username="user", password="user123"):
    return client.post("/login", data={"username": username, "password": password}, follow_redirects=True)


def test_index_and_login_available(app_instance):
    client = app_instance.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/login").status_code == 200


def test_authorization_and_admin_access_control(app_instance):
    client = app_instance.test_client()

    response = login(client)
    assert response.status_code == 200
    assert "Личный кабинет".encode("utf-8") in response.data

    response = client.get("/admin", follow_redirects=True)
    assert response.status_code == 200
    assert "Недостаточно прав".encode("utf-8") in response.data

    client.get("/logout")
    response = login(client, "admin", "admin123")
    assert response.status_code == 200
    response = client.get("/admin")
    assert response.status_code == 200
    assert "Административная панель".encode("utf-8") in response.data
