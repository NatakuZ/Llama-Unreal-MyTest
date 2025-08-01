// Copyright 2025-current Getnamo.

#include "LlamaComponent.h"  
#include "LlamaUtility.h"
#include "LlamaNative.h"
#include "Engine.h"  //engine
#include "Engine/World.h"  //engine
#include "Editor/EditorEngine.h" //unreal Ed 


// Definisci la classe qui
class FLlamaLatentAction : public FPendingLatentAction
{
public:
    FName ExecutionFunction;
    int32 OutputLink;
    FWeakObjectPtr CallbackTarget;
    FString* OutputResult;
    bool bCompleted = false;
	bool bSuccess = false;
    FLlamaLatentAction(const FLatentActionInfo& LatentInfo, FString& Output)
        : ExecutionFunction(LatentInfo.ExecutionFunction)
        , OutputLink(LatentInfo.Linkage)
        , CallbackTarget(LatentInfo.CallbackTarget)
        , OutputResult(&Output)
    {
    }

    virtual void UpdateOperation(FLatentResponse& Response) override
    {
        if (bCompleted)
        {
            Response.FinishAndTriggerIf(true, ExecutionFunction, OutputLink, CallbackTarget);
        }
    }

    void Complete(const FString& Result,bool Success)
    {
        if (OutputResult) *OutputResult = Result;
        bCompleted = true;
		bSuccess = Success;

    }
};

ULlamaComponent::ULlamaComponent(const FObjectInitializer &ObjectInitializer)
    : UActorComponent(ObjectInitializer)
{
    LlamaNative = new FLlamaNative();

    //Hookup native callbacks
    LlamaNative->OnModelStateChanged = [this](const FLLMModelState& UpdatedModelState)
    {
        ModelState = UpdatedModelState;
    };

    LlamaNative->OnTokenGenerated = [this](const FString& Token)
    {
        OnTokenGenerated.Broadcast(Token);
    };

    LlamaNative->OnResponseGenerated = [this](const FString& Response)
    {
        OnResponseGenerated.Broadcast(Response);
        OnEndOfStream.Broadcast(true, ModelState.LastTokenGenerationSpeed);
    };

	LlamaNative->OnResponseGeneratedWithStatus = [this](const FString& Response, ELLMResponseStatus Status)
	{
			//Emit response generated to general listeners
			OnResponseGeneratedWithStatus.Broadcast(Response, Status);
			OnEndOfStream.Broadcast(true, ModelState.LastTokenGenerationSpeed);
	};

    LlamaNative->OnPartialGenerated = [this](const FString& Partial)
    {
        OnPartialGenerated.Broadcast(Partial);
    };
    LlamaNative->OnPromptProcessed = [this](int32 TokensProcessed, EChatTemplateRole Role, float Speed)
    {
        OnPromptProcessed.Broadcast(TokensProcessed, Role, Speed);
    };
    LlamaNative->OnError = [this](const FString& ErrorMessage, int32 ErrorCode)
    {
        OnError.Broadcast(ErrorMessage, ErrorCode);
    };

    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.bStartWithTickEnabled = true;

    //All sentence ending formatting.
    ModelParams.Advanced.PartialsSeparators.Add(TEXT("."));
    ModelParams.Advanced.PartialsSeparators.Add(TEXT("?"));
    ModelParams.Advanced.PartialsSeparators.Add(TEXT("!"));
}
void ULlamaComponent::BeginPlay() 
{   
Super::BeginPlay();
//Initialize the native component
UE_LOG(LogTemp, Warning, TEXT("ULlamaComponent: BeginPlay, Owner=%s, World=%s"),
    GetOwner() ? *GetOwner()->GetName() : TEXT("NULL"),
    GetWorld() ? *GetWorld()->GetName() : TEXT("NULL"));


}
ULlamaComponent::~ULlamaComponent()
{
	if (LlamaNative)
	{
		delete LlamaNative;
		LlamaNative = nullptr;
	}
}

void ULlamaComponent::Activate(bool bReset)
{
    Super::Activate(bReset);

    if (ModelParams.bAutoLoadModelOnStartup)
    {
        LoadModel(true);
    }
}

void ULlamaComponent::Deactivate()
{
    Super::Deactivate();
}

void ULlamaComponent::TickComponent(float DeltaTime,
                                    ELevelTick TickType,
                                    FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    //Forward tick to llama so it can process the game thread callbacks
    LlamaNative->OnGameThreadTick(DeltaTime);
}

void ULlamaComponent::InsertTemplatedPrompt(const FString& Text, EChatTemplateRole Role, bool bAddAssistantBOS, bool bGenerateReply)
{
    FLlamaChatPrompt ChatPrompt;
    ChatPrompt.Prompt = Text;
    ChatPrompt.Role = Role;
    ChatPrompt.bAddAssistantBOS = bAddAssistantBOS;
    ChatPrompt.bGenerateReply = bGenerateReply;
    InsertTemplatedPromptStruct(ChatPrompt);
}

void ULlamaComponent::InsertTemplatedPromptStruct(const FLlamaChatPrompt& ChatPrompt)
{
    LlamaNative->InsertTemplatedPrompt(ChatPrompt);/*, [this, ChatPrompt](const FString& Response));
     {
        if (ChatPrompt.bGenerateReply)
        {
            OnResponseGenerated.Broadcast(Response);
            OnEndOfStream.Broadcast(true, ModelState.LastTokenGenerationSpeed);
        }
    });*/
}

void ULlamaComponent::InsertRawPrompt(const FString& Text, bool bGenerateReply)
{
    LlamaNative->InsertRawPrompt(Text, bGenerateReply); /*, [this, bGenerateReply](const FString& Response)
    {
        if (bGenerateReply)
        {
            OnResponseGenerated.Broadcast(Response);
            OnEndOfStream.Broadcast(true, ModelState.LastTokenGenerationSpeed);
        }
    });*/
}

void ULlamaComponent::LoadModel(bool bForceReload)
{

    LlamaNative->SetModelParams(ModelParams);
    LlamaNative->SetPooling(PoolingMode,PoolingType);
    LlamaNative->LoadModel(bForceReload, [this](const FString& ModelPath, int32 StatusCode)
    {
        //We errored, the emit will happen before we reach here so just exit
        if (StatusCode !=0)
        {
            return;
        }

        OnModelLoaded.Broadcast(ModelPath);
    });
}

void ULlamaComponent::UnloadModel()
{
    LlamaNative->UnloadModel([this](int32 StatusCode)
    {
        //this pretty much should never get called, just in case: emit.
        if (StatusCode != 0)
        {
            FString ErrorMessage = FString::Printf(TEXT("UnloadModel returned error code: %d"), StatusCode);
            UE_LOG(LlamaLog, Warning, TEXT("%s"), *ErrorMessage);
            OnError.Broadcast(ErrorMessage, StatusCode);
        }
    });
}

bool ULlamaComponent::IsModelLoaded()
{
    return ModelState.bModelIsLoaded;
}

void ULlamaComponent::ResetContextHistory(bool bKeepSystemPrompt)
{
    LlamaNative->ResetContextHistory(bKeepSystemPrompt);
}

void ULlamaComponent::RemoveLastAssistantReply()
{
    if (ModelParams.bRemoteMode)
    {
        //modify state only
        int32 Count = ModelState.ChatHistory.History.Num();
        if (Count >0)
        {
            ModelState.ChatHistory.History.RemoveAt(Count - 1);
        }
    }
    else
    {
        LlamaNative->RemoveLastReply();
    }
}

void ULlamaComponent::RemoveLastUserInput()
{
    if (ModelParams.bRemoteMode)
    {
        //modify state only
        int32 Count = ModelState.ChatHistory.History.Num();
        if (Count > 1)
        {
            ModelState.ChatHistory.History.RemoveAt(Count - 1);
            ModelState.ChatHistory.History.RemoveAt(Count - 2);
        }
    }
    else
    {
        LlamaNative->RemoveLastUserInput();
    }
}


void ULlamaComponent::ImpersonateTemplatedPrompt(const FLlamaChatPrompt& ChatPrompt)
{
    LlamaNative->SetModelParams(ModelParams);

    LlamaNative->ImpersonateTemplatedPrompt(ChatPrompt);
}

void ULlamaComponent::ImpersonateTemplatedToken(const FString& Token, EChatTemplateRole Role, bool bEoS)
{
    LlamaNative->ImpersonateTemplatedToken(Token, Role, bEoS);
}

FString ULlamaComponent::WrapPromptForRole(const FString& Text, EChatTemplateRole Role, const FString& Template)
{
    return LlamaNative->WrapPromptForRole(Text, Role, Template);
}

void ULlamaComponent::StopGeneration()
{
    LlamaNative->StopGeneration();
}

void ULlamaComponent::ResumeGeneration()
{
    LlamaNative->ResumeGeneration();
}

FString ULlamaComponent::RawContextHistory()
{
    return ModelState.ContextHistory;
}

FStructuredChatHistory ULlamaComponent::GetStructuredChatHistory()
{
    return ModelState.ChatHistory;
}

void ULlamaComponent::GeneratePromptEmbeddingsForText(const FString& Text)
{
    if (!ModelParams.Advanced.bEmbeddingMode)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model is not in embedding mode, cannot generate embeddings."));
        return;
    }

    LlamaNative->GetPromptEmbeddings(Text, [this](const TArray<float>& Embeddings, const FString& SourceText)
    {
        OnEmbeddings.Broadcast(Embeddings, SourceText);
    });
}

void ULlamaComponent::ConvertJson(const FString& Input, const FString& Output)
{
    LlamaNative->ConvertJson(Input,Output);
}



FString ULlamaComponent::RetriveFromEmbedding(const FString& Text)
{
    FString AugmentedText;
    AugmentedText = LlamaNative->RetriveFromEmbedding(Text);
    return AugmentedText+Text;
}


//ULlamaComponent* ULlamaComponent::RetriveFromJsonAsync(UObject* WorldContext, const FString& Text, const FString& Json, int NChuncksOut)
//{
//    ULlamaComponent* BlueprintNode = NewObject<ULlamaComponent>();
//    BlueprintNode->WorldContext = WorldContext;
//	BlueprintNode->World = WorldContext->GetWorld();
//    return BlueprintNode;
//}

void ULlamaComponent::RetriveFromJsonAsync(FLatentActionInfo LatentInfo, const FString& Json, int NChuncksOut, const FString& Input, FString& Output)
{
	// Log the owner and component state
        AActor* Owner = GetOwner();
    UE_LOG(LogTemp, Warning, TEXT("ULlamaComponent: Owner=%s, IsRegistered=%d, IsActive=%d"),
        Owner ? *Owner->GetName() : TEXT("NULL"),
        IsRegistered(),
        IsActive());
    // Ottieni il world context
       World = GetWorld();

#if WITH_EDITOR
        if (!World && GEditor)
        {
            World = GEditor->GetEditorWorldContext().World();
        }
#endif

        if (!World)
        {
            UE_LOG(LlamaLog, Error, TEXT("RetriveFromJsonAsync: World context is null, cannot execute latent action."));
            FLlamaLatentAction* NewAction = new FLlamaLatentAction(LatentInfo, Output);
            NewAction->Complete(TEXT("Error: World context is null."), false);
            return;
        }

        FLatentActionManager& LatentManager = World->GetLatentActionManager();

        if (LatentManager.FindExistingAction<FLlamaLatentAction>(LatentInfo.CallbackTarget, LatentInfo.UUID) == nullptr)
        {
            FLlamaLatentAction* NewAction = new FLlamaLatentAction(LatentInfo, Output);

            AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [this, Json, NChuncksOut, Input, NewAction]()
                {
                    FString Result = RetriveFromJson(Input, Json, NChuncksOut);
                    FString TotalText = "<document>" + Result + "</document>\n" + "<q>" + Input + "</q>";
                    AsyncTask(ENamedThreads::GameThread, [NewAction, TotalText]()
                        {
                            NewAction->Complete(TotalText, true);
                        });
                });

            LatentManager.AddNewAction(LatentInfo.CallbackTarget, LatentInfo.UUID, NewAction);
        }
    

}

FString ULlamaComponent::RetriveFromJson(const FString& Text, const FString& Json, int NChuncksOut)
{
	
    FString AugmentedText;
    AugmentedText = LlamaNative->RetriveFromJson(Text,Json,NChuncksOut);
    FString TotalText="<document>"+AugmentedText+"</document>\n"+"<q>"+Text+"</q>";
    return TotalText;
}

bool ULlamaComponent::CheckContext()
{
    return LlamaNative->CheckContext();
}

void ULlamaComponent::BuildAndSaveIndexFromChunks(const TArray<FString>& TextChunks, const FString& IndexSavePath, const FString& MapSavePath)
{
    LlamaNative->BuildAndSaveIndexFromChunks(TextChunks, IndexSavePath, MapSavePath);
}

FString ULlamaComponent::FindNearestString(FString Query)
{
    return LlamaNative->FindNearestString(Query);
}

