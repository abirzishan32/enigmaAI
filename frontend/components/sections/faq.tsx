"use client"

import { useState } from "react"
import { Plus, Minus } from "lucide-react"

interface FAQItem {
  question: string
  answer: string
}

const faqData: FAQItem[] = [
  {
    question: "What is EnigmaAI?",
    answer: "EnigmaAI is an industry-first privacy-preserving AI framework that combines Fully Homomorphic Encryption (FHE) and attribute shuffling to protect sensitive data during LLM agent interactions. It enables secure computation on encrypted data, ensuring that your private information remains confidential even while being processed by AI models."
  },
  {
    question: "What are the elements of EnigmaAI?",
    answer: "EnigmaAI consists of three main components: (1) Homomorphic Encryption Layer - using TenSEAL for encrypting and processing data without decryption, (2) Attribute Shuffling Module - randomizing data attributes to prevent pattern recognition while maintaining computational accuracy, and (3) Secure LLM Integration - enabling AI agents to work with encrypted data seamlessly while maintaining 128-bit security standards."
  },
  {
    question: "What can be done with EnigmaAI?",
    answer: "EnigmaAI enables a wide range of privacy-preserving applications including: secure digit recognition and handwriting analysis, confidential healthcare data processing, private financial analysis and fraud detection, GDPR-compliant customer data analytics, and secure multi-party computation for collaborative AI tasks. All operations maintain data privacy throughout the entire computational pipeline."
  },
  {
    question: "How does privacy actually work?",
    answer: "EnigmaAI implements multiple layers of privacy protection. First, your data is encrypted using Fully Homomorphic Encryption (FHE) before leaving your device - this means computations can be performed on the encrypted data without ever decrypting it. Second, attribute shuffling adds an additional layer by randomizing data patterns, making it impossible to identify specific attributes even if the encryption were compromised. The result is that your sensitive data remains encrypted end-to-end, with zero-knowledge computation ensuring that neither the server nor any third party can access your raw data."
  }
]

function FAQItemComponent({ item, isOpen, onToggle }: { item: FAQItem; isOpen: boolean; onToggle: () => void }) {
  return (
    <div className="border-b border-border last:border-b-0">
      <button
        onClick={onToggle}
        className="w-full flex items-start justify-between gap-4 py-6 text-left hover:opacity-80 transition-opacity"
        aria-expanded={isOpen}
      >
        <h3 className="text-lg font-semibold text-foreground pr-8">
          {item.question}
        </h3>
        <div className="flex-shrink-0 mt-1">
          {isOpen ? (
            <Minus className="h-5 w-5 text-primary" strokeWidth={2} />
          ) : (
            <Plus className="h-5 w-5 text-muted-foreground" strokeWidth={2} />
          )}
        </div>
      </button>
      
      <div
        className={`overflow-hidden transition-all duration-300 ease-in-out ${
          isOpen ? "max-h-96 opacity-100 mb-6" : "max-h-0 opacity-0"
        }`}
      >
        <p className="text-muted-foreground leading-relaxed pr-12">
          {item.answer}
        </p>
      </div>
    </div>
  )
}

export function FAQSection() {
  const [openIndex, setOpenIndex] = useState<number | null>(null)

  const handleToggle = (index: number) => {
    setOpenIndex(openIndex === index ? null : index)
  }

  return (
    <section id="faq" className="py-24 md:py-32 bg-muted/30">
      <div className="container-custom">
        <div className="max-w-4xl mx-auto">
          {/* Section Header */}
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Frequently Asked Questions
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Everything you need to know about EnigmaAI and how it protects your data
            </p>
          </div>

          {/* FAQ Items */}
          <div className="bg-background rounded-lg border border-border p-2 md:p-4">
            {faqData.map((item, index) => (
              <FAQItemComponent
                key={index}
                item={item}
                isOpen={openIndex === index}
                onToggle={() => handleToggle(index)}
              />
            ))}
          </div>

          {/* Additional Help */}
          <div className="mt-12 text-center">
            <p className="text-muted-foreground mb-4">
              Still have questions?
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a 
                href="https://github.com" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-primary hover:underline font-medium"
              >
                View Documentation
              </a>
              <span className="hidden sm:inline text-muted-foreground">•</span>
              <a 
                href="https://github.com" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-primary hover:underline font-medium"
              >
                Join our GitHub Discussion
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
