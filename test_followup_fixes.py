#!/usr/bin/env python3
"""
Test follow-up question fixes:
1. Follow-ups are language-aware (not hardcoded English)
2. Follow-ups are not redundant (check if response already has question)
3. Conversation history is used to detect if user is confirming a prior question
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from pipeline.RAG import RAGPipeline
import json

def test_followup_language_awareness():
    """Test that follow-ups appear in the user's language"""
    print("\n" + "="*60)
    print("TEST 1: Follow-up Language Awareness")
    print("="*60)
    
    rag = RAGPipeline()
    
    # Simulate Javanese response without question mark
    test_response = {
        'jv': {
            'response': 'Jawa sik iku cara paling cetha kanggo njaga paspor amu',
            'expected_followup': 'Apa kowe perlu bantuan kanggo langkah sabanjure?',
            'should_contain': 'sabanjure'
        },
        'id': {
            'response': 'Bahasa Indonesia adalah cara terbaik untuk melindungi paspor Anda',
            'expected_followup': 'Apakah anda memerlukan bantuan untuk langkah berikutnya?',
            'should_contain': 'berikutnya'
        },
        'ms': {
            'response': 'Bahasa Melayu adalah cara terbaik untuk melindungi pasport anda',
            'expected_followup': 'Adakah anda memerlukan bantuan untuk langkah seterusnya?',
            'should_contain': 'seterusnya'
        },
        'tl': {
            'response': 'Ang Tagalog ay ang pinakamahusay na paraan upang protektahan ang iyong pasaporte',
            'expected_followup': 'Kailangan mo ba ng tulong para sa susunod na hakbang?',
            'should_contain': 'susunod'
        }
    }
    
    for lang, test_data in test_response.items():
        result = rag.post_process_response(
            test_data['response'],
            language=lang,
            conversation_history=[]
        )
        answer = result['answer']
        
        print(f"\nLanguage: {lang.upper()}")
        print(f"Response: {test_data['response']}")
        print(f"Post-processed:\n{answer}")
        
        if test_data['should_contain'] in answer:
            print(f"✓ Follow-up is in {lang}!")
        else:
            print(f"✗ Follow-up NOT in {lang}. Expected to find: {test_data['should_contain']}")
            print(f"Full answer: {answer}")

def test_no_redundant_followups():
    """Test that LLM responses ending with ? don't get another question added"""
    print("\n" + "="*60)
    print("TEST 2: No Redundant Follow-ups (Response Already Ends with ?)")
    print("="*60)
    
    rag = RAGPipeline()
    
    llm_response_with_question = "Here are the steps:\n1. Fill out form\n2. Submit\nWould you like a checklist to print?"
    
    result = rag.post_process_response(
        llm_response_with_question,
        language='en',
        conversation_history=[]
    )
    
    answer = result['answer']
    print(f"\nLLM Response:\n{llm_response_with_question}")
    print(f"\nPost-processed:\n{answer}")
    
    # Count how many question marks
    question_count = answer.count('?')
    print(f"\nQuestion mark count: {question_count}")
    
    if question_count == 1:
        print("✓ No redundant question added!")
    else:
        print(f"✗ Redundant question detected! Found {question_count} questions instead of 1.")

def test_conversation_context_awareness():
    """Test that follow-ups are skipped when user is confirming a prior yes/no question"""
    print("\n" + "="*60)
    print("TEST 3: Conversation Context Awareness")
    print("="*60)
    
    rag = RAGPipeline()
    
    # Simulate conversation: assistant asks question, user confirms
    conversation_with_question = [
        {
            'role': 'assistant',
            'text': 'Do you want detailed steps or just a summary?'
        },
        {
            'role': 'user',
            'text': 'Yes, detailed steps please'
        }
    ]
    
    llm_response = "Here are the detailed steps:\n1. Step one\n2. Step two"
    
    result = rag.post_process_response(
        llm_response,
        language='en',
        user_query='Yes, detailed steps please',
        conversation_history=conversation_with_question
    )
    
    answer = result['answer']
    print(f"\nPrior Assistant Question: {conversation_with_question[0]['text']}")
    print(f"User Confirmation: {conversation_with_question[1]['text']}")
    print(f"LLM Response: {llm_response}")
    print(f"\nPost-processed:\n{answer}")
    
    # Check if a follow-up was added
    has_extra_followup = 'Would you like help' in answer or 'perlu bantuan' in answer
    
    if not has_extra_followup:
        print("\n✓ No redundant follow-up after user confirmed!")
    else:
        print("\n✗ Follow-up was added even though user was confirming prior question")

def test_language_agnostic_step_gate():
    """Test that [STEP_GATE] prevents follow-ups regardless of language"""
    print("\n" + "="*60)
    print("TEST 4: [STEP_GATE] Prevents Follow-ups in Any Language")
    print("="*60)
    
    rag = RAGPipeline()
    
    response_with_gate = "Step 1: Do something\n[STEP_GATE]\nStep 2: Do next"
    
    result = rag.post_process_response(
        response_with_gate,
        language='jv',  # Javanese
        conversation_history=[]
    )
    
    answer = result['answer']
    print(f"\nResponse with [STEP_GATE]: {response_with_gate}")
    print(f"\nPost-processed:\n{answer}")
    
    # Should NOT have question added despite no trailing ?
    has_followup = 'sambanjure' in answer or 'perlu bantuan' in answer
    
    if not has_followup:
        print("\n✓ [STEP_GATE] successfully prevents follow-up!")
    else:
        print("\n✗ Follow-up was added despite [STEP_GATE]")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTING FOLLOW-UP QUESTION FIXES")
    print("="*60)
    
    try:
        test_followup_language_awareness()
        test_no_redundant_followups()
        test_conversation_context_awareness()
        test_language_agnostic_step_gate()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
