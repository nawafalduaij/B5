### Analogy: The Hotel Key Card

Think of an API Key as a **hotel key card**.

* **The Hotel (Server):** Has resources (rooms) but keeps them locked.
* **The Guest (Client):** Wants access.
* **The Key Card (API Key):** Identifies you and proves you are allowed to enter specific rooms.

---

### What & Why

An API key is a unique string of characters used to identify the calling program.

* **Identification:** Keys "authenticate the calling project," allowing the server to recognize who is asking for data.
* **Control:** This lets the server track usage for billing and enforce limits (quotas) so one user doesn't crash the system.

---

### How to Use It

You send the key with your request so the server accepts it. This is done in two ways:

1. **Query String:** Added to the end of the URL (e.g., `?apikey=12345`).
2. **Header (Best Practice):** Sent in the request metadata (e.g., `Authorization: 12345`), keeping it out of the browser address bar.

---

### Security Risks

If you lose your key, it is like dropping your credit card.

* **Theft:** Attackers can use your key to make requests on your behalf.
* **Consequences:** You suffer **financial loss** (paying for their usage) or **service denial** (they use up your available quota).

> **Rule:** Never post keys on public sites like GitHub.