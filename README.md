Render Vectorization API. 
-
This code is purposed to be uploaded on a Render server wich has a persist disc. 

It has two main purposes:

-------------------------
The main file contains the main logic for the vectorization process. It has all the essentioal methods for operations. 
It uses ChromaDB for the vectorization processes and FastAPI for the calls. 
________________________

The vectorization_check file contains logic that saves all the agents that have passed the checking process. 
The process can check if the needed update type is hourly, daily or monthly update.

-----------------------
