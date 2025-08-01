// Copyright 2025-current Getnamo.

#include "Embedding/VectorDatabase.h"
#include "Misc/Paths.h"
#include "hnswlib/hnswlib.h"

#include "LlamaUtility.h"

class FHNSWPrivate
{
public:
    hnswlib::HierarchicalNSW<float>* HNSW = nullptr;

    void InitializeHNSW(const FVectorDBParams& Params)
    {
        ReleaseHNSWIfAllocated();
        hnswlib::L2Space Space(Params.Dimensions);
        HNSW = new hnswlib::HierarchicalNSW<float>(&Space, Params.MaxElements, Params.M, Params.EFConstruction);

      
    }

    void ReleaseHNSWIfAllocated()
    {
        UE_LOG(LogTemp, Log, TEXT("ReleaseHNSWIfAllocated begin "));
        if (HNSW)
        {
            UE_LOG(LogTemp, Log, TEXT("ReleaseHNSWIfAllocated HNSW "));
            delete HNSW;
            HNSW = nullptr;
        }
        UE_LOG(LogTemp, Log, TEXT("ReleaseHNSWIfAllocated end "));
    }
    ~FHNSWPrivate()
    {
        ReleaseHNSWIfAllocated();
    }
};

void FVectorDatabase::BasicsTest()
{
    //Try: https://github.com/nmslib/hnswlib/blob/master/examples/cpp/EXAMPLES.md

    InitializeDB();

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[Params.Dimensions * Params.MaxElements];
    for (int i = 0; i < Params.Dimensions * Params.MaxElements; i++)
    {
        data[i] = distrib_real(rng);
    }

    // Add data to index
    
    for (int i = 0; i < Params.MaxElements; i++)
    {
        Private->HNSW->addPoint(data + i * Params.Dimensions, i);
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < Params.MaxElements; i++)
    {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = Private->HNSW->searchKnn(data + i * Params.Dimensions, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    float recall = correct / Params.MaxElements;

    UE_LOG(LogTemp, Log, TEXT("Recall: %1.3f"), recall);

    // Serialize index
    FString SavePath = FPaths::ProjectSavedDir() / TEXT("hnsw.bin");
    std::string HNSWPath = FLlamaString::ToStd(SavePath);
    Private->HNSW->saveIndex(HNSWPath);
    delete Private->HNSW;

    // Deserialize index and check recall
    // This test appears to fail in unreal context (loading index)
    hnswlib::L2Space Space(Params.Dimensions);
    Private->HNSW = new hnswlib::HierarchicalNSW<float>(&Space, HNSWPath, false, Params.MaxElements);
    //HNSW = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    //HNSW->loadIndex(HNSWPath, &space);

    if (Private->HNSW->getMaxElements() > 0)
    {
        correct = 0;
        for (int i = 0; i < Private->HNSW->getMaxElements(); i++)
        {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = Private->HNSW->searchKnn(data + i * Params.Dimensions, 1);
            hnswlib::labeltype label = result.top().second;
            if (label == i) correct++;
        }
        recall = (float)correct / Params.MaxElements;
        UE_LOG(LogTemp, Log, TEXT("Recall of deserialized index: %1.3f"), recall);
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("Failed to load index from file correctly"));
    }
    delete[] data;
    //delete HNSW; //handled at deconstructor atm
}

void FVectorDatabase::InitializeDB()
{
    //Delete and re-initialize as needed
	TextDatabaseArray.Empty();
    Private->InitializeHNSW(Params);
    FString SavePath = FPaths::ProjectSavedDir() / TEXT("hnsw.bin");
    std::string HNSWPath = FLlamaString::ToStd(SavePath);
    Private->HNSW->saveIndex(HNSWPath);
    delete Private->HNSW;

    // Deserialize index and check recall
    // This test appears to fail in unreal context (loading index)
    hnswlib::L2Space Space(Params.Dimensions);
    Private->HNSW = new hnswlib::HierarchicalNSW<float>(&Space, HNSWPath, false, Params.MaxElements);

}

void FVectorDatabase::AddVectorEmbeddingIdPair(const TArray<float>& Embedding, int64 UniqueId)
{
   
    size_t IndexDimension = Private->HNSW->data_size_;
    IndexDimension = IndexDimension / sizeof(float);
    if (Embedding.Num() != IndexDimension)
    {
        UE_LOG(LogTemp, Error, TEXT("Dimensione dell'embedding (%d) non corrisponde alla dimensione dell'indice (%d)!"), Embedding.Num(), IndexDimension);
     //   return;
    }
    Private->HNSW->addPoint(&Embedding, UniqueId);
}

void FVectorDatabase::AddVectorEmbeddingStringPair(const TArray<float>& Embedding, const FString& Text)
{
	TextDatabaseArray.Empty(); // Pulisci il database di testo prima di aggiungere un nuovo elemento    

   int64 UniqueId = TextDatabaseMaxId;
   // TextDatabase.Add(UniqueId, Text);
    UE_LOG(LogTemp, Log, TEXT("TextDatabaseArray.Num() prima di Add: %d"), TextDatabaseArray.Num());
    TextDatabaseArray.Add(Text);
   UE_LOG(LogTemp, Log, TEXT("TextDatabaseArray.Num() dopo Add: %d"), TextDatabaseArray.Num());

	UE_LOG(LogTemp, Log, TEXT("Aggiunta di un nuovo elemento al database vettoriale con ID unico: %lld"), UniqueId);

    AddVectorEmbeddingIdPair(Embedding, UniqueId);
	UE_LOG(LogTemp, Log, TEXT("Aggiunta di un nuovo elemento al database vettoriale con embedding di dimensione %d e ID unico: %lld"), Embedding.Num(), UniqueId);
    TextDatabaseMaxId++;
}

int64 FVectorDatabase::FindNearestId(const TArray<float>& ForEmbedding, std::vector<float>& EmbArray)
{
    TArray<int64> Ids;
    FindNearestNIds(Ids, ForEmbedding, EmbArray, 1);
    if (Ids.Num() > 0)
    {
        return Ids[0];
    }
    else
    {
        return -1;
    }
}

FString FVectorDatabase::FindNearestString(const TArray<float>& ForEmbedding,std::vector<float>& EmbArray)
{
    TArray<FString> StringResults;
    FindNearestNStrings(StringResults, ForEmbedding, EmbArray, 1);

    if (StringResults.Num()>0)
    {
        return StringResults[0];
    }
    else
    {
        return TEXT("");
    }
}

void FVectorDatabase::FindNearestNIds(TArray<int64>& IdResults, const TArray<float>& ForEmbedding,std::vector<float>& EmbArray, int32 N)
{
    /*
    std::priority_queue<std::pair<float, hnswlib::labeltype>> Results = Private->HNSW->searchKnn(EmbArray.data(), N);

    while (!Results.empty())
    {
        const std::pair<float, hnswlib::labeltype>& Item = Results.top();
        IdResults.Add(static_cast<int64>(Item.second));
        Results.pop();
    }
    */
        IdResults.Empty(); // Pulisci i risultati precedenti
        if (!Private->HNSW) {
            hnswlib::L2Space Space(Params.Dimensions);
            Private->HNSW = new hnswlib::HierarchicalNSW<float>(&Space, Params.MaxElements, Params.M, Params.EFConstruction);
            UE_LOG(LogTemp, Fatal, TEXT("rinizializato"));

        }

        // --- INIZIO CONTROLLI DI SICUREZZA ---

        // CONTROLLO 1: L'oggetto HNSW esiste?
        if (!Private || !Private->HNSW)
        {
            UE_LOG(LogTemp, Fatal, TEXT("CRASH IMMINENTE: Il database vettoriale (HNSW) non è stato inizializzato. Hai chiamato InitializeDB()?"));
            return;
        }

        // CONTROLLO 2: Il database contiene degli elementi?
        if (Private->HNSW->getCurrentElementCount() == 0)
        {
            UE_LOG(LogTemp, Warning, TEXT("Ricerca eseguita su un database vuoto. Nessun risultato possibile."));
            return;
        }

        // CONTROLLO 3: L'embedding della domanda è valido?
        if (ForEmbedding.Num() == 0)
        {
            UE_LOG(LogTemp, Error, TEXT("CRASH IMMINENTE: Stai cercando di fare una ricerca con un embedding di domanda vuoto."));
            return;
        }

        // CONTROLLO 4: Le dimensioni corrispondono?
        const size_t IndexDimension = Private->HNSW->data_size_ / sizeof(float);
        if (ForEmbedding.Num() != IndexDimension)
        {
            UE_LOG(LogTemp, Warning, TEXT("CRASH IMMINENTE: La dimensione dell'embedding della domanda (%d) non corrisponde alla dimensione dell'indice (%zu)!"), ForEmbedding.Num(), IndexDimension);
        //    return;
        }  

        UE_LOG(LogTemp, Log, TEXT("Tutti i controlli preliminari superati. Inizio la ricerca con HNSWlib..."));
        // Validazione dimensioni embedding
        if (ForEmbedding.Num() != Params.Dimensions)
        {
            UE_LOG(LogTemp, Error, TEXT("Dimensione embedding non valida: atteso %d, ricevuto %d"), Params.Dimensions, ForEmbedding.Num());
            return;
		}
		else {
			UE_LOG(LogTemp, Log, TEXT("Dimensione embedding valida: %d"), ForEmbedding.Num());
		}

        // Se usi anche EmbArray, valida anche quella
        if (EmbArray.size() != static_cast<size_t>(Params.Dimensions))
        {
            UE_LOG(LogTemp, Error, TEXT("Dimensione EmbArray non valida: atteso %d, ricevuto %zu"), Params.Dimensions, EmbArray.size());
            return;
		}
		else {
			UE_LOG(LogTemp, Log, TEXT("Dimensione EmbArray valida: %zu"), EmbArray.size());
		}
        // --- FINE CONTROLLI DI SICUREZZA ---
        size_t expectedDim = Params.Dimensions; // oppure recupera da HNSW->data_size_
        if (EmbArray.size() != expectedDim) {
            UE_LOG(LogTemp, Error, TEXT("Dimensione embedding non valida: atteso %zu, ricevuto %zu"), expectedDim, EmbArray.size());
            return; // oppure gestisci l’errore come preferisci
        }
		else
		{
			UE_LOG(LogTemp, Log, TEXT("Dimensione embedding valida: %zu"), EmbArray.size());
		}

        // Chiamata a HNSWlib incapsulata in un blocco try-catch per "catturare" eventuali errori interni
       // try
        //{
            UE_LOG(LogTemp, Log, TEXT("funzione searchknn"));
            std::priority_queue<std::pair<float, hnswlib::labeltype>> Results = Private->HNSW->searchKnn(&ForEmbedding, N);

            while (!Results.empty())
            {
                UE_LOG(LogTemp, Log, TEXT("funzione result"));
                const std::pair<float, hnswlib::labeltype>& Item = Results.top();
                IdResults.Add(static_cast<int64>(Item.second));
                Results.pop();
            }
            UE_LOG(LogTemp, Log, TEXT("after while  result"));
            // Inverti l'array per avere i risultati dal più vicino al più lontano
            Algo::Reverse(IdResults);

            UE_LOG(LogTemp, Log, TEXT("Ricerca completata. Trovati %d risultati."), IdResults.Num());
 /*       }
        catch (const std::exception& e)
        {
             Se HNSWlib lancia un'eccezione, la catturiamo qui invece di far crashare tutto
            UE_LOG(LogTemp, Fatal, TEXT("HNSWLib ha lanciato un'eccezione durante searchKnn: %s"), UTF8_TO_TCHAR(e.what()));
        }*/
    
}

void FVectorDatabase::FindNearestNStrings(TArray<FString>& StringResults, const TArray<float>& ForEmbedding, std::vector<float>& EmbArray,int32 N)
{
    TArray<int64> Ids;
    FindNearestNIds(Ids, ForEmbedding, EmbArray, N);

    for (int64 Id : Ids)
    {
        FString* MaybeResult = TextDatabase.Find(Id);
        if (MaybeResult)
        {
            FString StringResult = *MaybeResult;
            StringResults.Add(StringResult);
        }
    }
}

FVectorDatabase::FVectorDatabase()
{
    Private = new FHNSWPrivate();

    //llamainternal
    //1. Load model
    //2. Tokenize input
    //3. llama_model_n_embd & embeddings for input allocation
    //4. batch_decode
    
    //1-4 now works within llama internal & native. 

    //5. store embeddings in index.
    //6. potentially save index

    //we should store and retrieve stored embeddings
    //What's a good api for VectorDB?
}

FVectorDatabase::~FVectorDatabase()
{
    TextDatabase.Empty();
    delete Private;
    Private = nullptr;
}

bool FVectorDatabase::SaveIndex(const FString& IndexFilePath, const FString& MapFilePath)
{

    // Serialize index
    FString SavePath = FPaths::ProjectSavedDir() + IndexFilePath;
    std::string HNSWPath = FLlamaString::ToStd(SavePath);
    Private->HNSW->saveIndex(HNSWPath);
    delete Private->HNSW;
    return true;
}

bool FVectorDatabase::LoadIndex(const FString& IndexFilePath, const FString& MapFilePath)
{
    // --- CONTROLLI PRELIMINARI ---
    if (!FPaths::FileExists(IndexFilePath) || !FPaths::FileExists(MapFilePath))
    {
        UE_LOG(LogTemp, Warning, TEXT("LoadIndex: Uno o entrambi i file di database non esistono."));
        return false;
    }

    // --- PARTE 1: CARICARE LA MAPPA ID -> TESTO DAL FILE JSON ---
    UE_LOG(LogTemp, Log, TEXT("Caricamento della mappa Testo da: %s"), *MapFilePath);

    // Leggi il file JSON in una stringa
    FString JsonString;
    if (!FFileHelper::LoadFileToString(JsonString, *MapFilePath))
    {
        UE_LOG(LogTemp, Error, TEXT("Impossibile leggere il file della mappa JSON: %s"), *MapFilePath);
        return false;
    }

    // Fai il parsing della stringa JSON
    TSharedPtr<FJsonObject> RootObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);
    if (!FJsonSerializer::Deserialize(Reader, RootObject) || !RootObject.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("Impossibile fare il parsing del file della mappa JSON. Controlla che il formato sia corretto."));
        return false;
    }

    // Svuota la mappa corrente e riempila con i dati del file
    TextDatabase.Empty();
    int64 MaxId = 0;
    for (const auto& Pair : RootObject->Values)
    {
        // La chiave nel JSON è una stringa, la riconvertiamo in int64
        int64 Key = FCString::Atoi64(*Pair.Key);
        FString Value = Pair.Value->AsString();

        TextDatabase.Add(Key, Value);

        // Teniamo traccia dell'ID più alto per le prossime aggiunte
        if (Key > MaxId)
        {
            MaxId = Key;
        }
    }
    TextDatabaseMaxId = MaxId;
    UE_LOG(LogTemp, Log, TEXT("Mappa Testo caricata con successo. Trovati %d elementi."), TextDatabase.Num());


    // --- PARTE 2: CARICARE L'INDICE HNSW DAL FILE BINARIO ---
    UE_LOG(LogTemp, Log, TEXT("Caricamento dell'indice HNSW da: %s"), *IndexFilePath);

    // Prima di caricare un nuovo indice, rilasciamo quello vecchio se esiste per evitare memory leak
    Private->ReleaseHNSWIfAllocated();

    // Avvolgiamo la chiamata alla libreria C++ esterna in un blocco try...catch
    // per gestire eventuali eccezioni che potrebbero causare un crash.
    // try
    //{
        // Controlla che i parametri essenziali siano stati impostati
        if (Params.Dimensions <= 0 || Params.MaxElements <= 0)
        {
            throw std::runtime_error("I parametri del DB (Dimensions, MaxElements) non sono stati inizializzati prima di caricare l'indice.");
        }

        // Crea lo "spazio metrico" (L2 = distanza euclidea) necessario a HNSWlib
        hnswlib::L2Space Space(Params.Dimensions);
        std::string HNSWPath = TCHAR_TO_UTF8(*IndexFilePath);

        // Questa è la funzione di HNSWlib che carica l'indice da file.
        // NOTA: Come segnalato nel codice di esempio che hai trovato, questa funzione a volte può essere instabile.
        Private->HNSW = new hnswlib::HierarchicalNSW<float>(&Space, HNSWPath, false, Params.MaxElements);

        // Controlla se il caricamento ha avuto un esito positivo
        if (Private->HNSW == nullptr || Private->HNSW->getCurrentElementCount() == 0)
        {
            throw std::runtime_error("L'indice HNSW è stato caricato ma risulta nullo o vuoto.");
        }
    //}
    //catch (const std::exception& e)
    //{
    //    UE_LOG(LogTemp, Error, TEXT("Errore critico durante il caricamento dell'indice HNSW: %s"), UTF8_TO_TCHAR(e.what()));
    //    Private->ReleaseHNSWIfAllocated(); // Pulisci in caso di fallimento
    //    TextDatabase.Empty(); // Svuota anche la mappa per mantenere uno stato consistente
    //    return false;
    //}

    UE_LOG(LogTemp, Log, TEXT("Indice HNSW caricato con successo con %zu elementi."), Private->HNSW->getCurrentElementCount());

    return true;
}
/*
void FVectorDatabase::BasicsTest()
{
    //Try: https://github.com/nmslib/hnswlib/blob/master/examples/cpp/EXAMPLES.md

    InitializeDB();

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[Params.Dimensions * Params.MaxElements];
    for (int i = 0; i < Params.Dimensions * Params.MaxElements; i++)
    {
        data[i] = distrib_real(rng);
    }

    // Add data to index
    for (int i = 0; i < Params.MaxElements; i++)
    {
        Private->HNSW->addPoint(data + i * Params.Dimensions, i);
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < Params.MaxElements; i++)
    {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = Private->HNSW->searchKnn(data + i * Params.Dimensions, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    float recall = correct / Params.MaxElements;

    UE_LOG(LogTemp, Log, TEXT("Recall: %1.3f"), recall);

    // Serialize index
    FString SavePath = FPaths::ProjectSavedDir() / TEXT("hnsw.bin");
    std::string HNSWPath = FLlamaString::ToStd(SavePath);
    Private->HNSW->saveIndex(HNSWPath);
    delete Private->HNSW;

    // Deserialize index and check recall
    // This test appears to fail in unreal context (loading index)
    hnswlib::L2Space Space(Params.Dimensions);
    Private->HNSW = new hnswlib::HierarchicalNSW<float>(&Space, HNSWPath, false, Params.MaxElements);
    //HNSW = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    //HNSW->loadIndex(HNSWPath, &space);

    if (Private->HNSW->getMaxElements() > 0)
    {
        correct = 0;
        for (int i = 0; i < Private->HNSW->getMaxElements(); i++)
        {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = Private->HNSW->searchKnn(data + i * Params.Dimensions, 1);
            hnswlib::labeltype label = result.top().second;
            if (label == i) correct++;
        }
        recall = (float)correct / Params.MaxElements;
        UE_LOG(LogTemp, Log, TEXT("Recall of deserialized index: %1.3f"), recall);
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("Failed to load index from file correctly"));
    }
    delete[] data;
    //delete HNSW; //handled at deconstructor atm
}
*/