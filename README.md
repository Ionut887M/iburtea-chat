# iburtea-chat
Pentru a rula proiectul: git clone <link-ul repository-ului>
Versiunea Python necesară: 3.9
După crearea venv-ului, rulați comanda pip install farm-haystack[inference] pentru a instala dependințele necesare
În cod, la linia 15 trebuie specificat path-ul din PC-ul dumneavoastră către setul de date.
Trebuie creat în directorul de lucru un fisier .env care să conțină HF_TOKEN = <TOKENUL HUGGINGFACE, CARE TREBUIE SĂ AIBĂ ACCES LA LLAMA-3>
Pentru a antrena și manipula AI-ul am ales să folosesc librăria Haystack v1.

La primul pas, am pregătit datele necesare, setul de date. Am folosit metoda convert_files_to_docs, funcție care convertește fișierele din directorul care conține setul de date în tipul necesar de date pentru Haystack.

După conversie, am creat un store în care vom stoca documentele obținute în urma conversiei.
Mai departe am definit o variabila retriever. Aceasta este defapt o funcție de ranking, ce facilitează recuperarea mai rapidă a documentelor mai relevante pentru interogare.

Mai departe am creat un PromptTemplate, pentru a structura inputul si outputul, in care specific formatul dat ce cerință.

Am creat PromptNode, pe baza template-ului descris mai devreme. Acesta va procesa interogarea și documentele adăugate in store, pentru a genera oferta detaliată.

Pentru a conecta retriever-ul cu nodul, voi folosi Pipeline, pentru a găsi cele mai relevante documente și pe baza acestora generează oferta.

Deoarece Haystack funcționează pe baza agenților, iar aceștia lucrează și gândesc iterativ, reluând răspunsul de mai multe ori până ajung la răspunsul final, am creat o unealtă de căutare care să fie folosită în această iterație.

La ultimul pas, am creat un prompt pentru agent, prin care îl învăț cum ar trebui să răspundă, ce ar trebui să spună, cum ar trebui sa genereze răspunsul, ce metode să folosească, de unde să își procure informațiile și care trebuie să fie forma finală a răspunsului.

