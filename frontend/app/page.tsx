import { HeroSection } from "@/components/sections/hero"
import { ProblemSection } from "@/components/sections/problem"
import { SolutionSection } from "@/components/sections/solution"
import { HowItWorksSection } from "@/components/sections/how-it-works"
import { FeaturesSection } from "@/components/sections/features"
import { TechnicalSpecsSection } from "@/components/sections/technical-specs"
import { UseCasesSection } from "@/components/sections/use-cases"
import { ComparisonSection } from "@/components/sections/comparison"
import { FAQSection } from "@/components/sections/faq"
import { FooterSection } from "@/components/sections/footer"
import {
  GlowingStarsBackgroundCard,
  GlowingStarsDescription,
  GlowingStarsTitle,
} from "@/components/ui/glowing-stars"

export default function Home() {
  return (
    <main className="min-h-screen">
      <HeroSection />
      <ProblemSection />
      <SolutionSection />
      <HowItWorksSection />
      <FeaturesSection />

      <TechnicalSpecsSection />
      <UseCasesSection />
      <ComparisonSection />


      <section className="py-24 bg-gradient-to-b from-background to-muted">
        <div className="container-custom">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Try it yourself
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Experience the power of privacy-preserving AI with our interactive demos
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
            {/* Digit Classification Card */}
            <a
              href="/digit-recog"
              className="group relative overflow-hidden rounded-2xl border border-border bg-card p-8 transition-all hover:shadow-premium hover:scale-[1.02] hover:border-primary/50"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="relative space-y-4">
                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
                  <svg className="w-6 h-6 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                  </svg>
                </div>
                <h3 className="text-xl font-semibold">Digit Classification</h3>
                <p className="text-muted-foreground text-sm">
                  Draw a digit and watch our FHE model classify it without ever seeing your raw data
                </p>
                <div className="flex items-center gap-2 text-primary text-sm font-medium pt-2">
                  Try Demo
                  <svg className="w-4 h-4 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </div>
              </div>
            </a>

            {/* Private Chatbot Card */}
            <a
              href="/chatbot"
              className="group relative overflow-hidden rounded-2xl border border-border bg-card p-8 transition-all hover:shadow-premium hover:scale-[1.02] hover:border-primary/50"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="relative space-y-4">
                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
                  <svg className="w-6 h-6 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                  </svg>
                </div>
                <h3 className="text-xl font-semibold">Private Chatbot</h3>
                <p className="text-muted-foreground text-sm">
                  Chat with an AI assistant that processes your queries using encrypted computation
                </p>
                <div className="flex items-center gap-2 text-primary text-sm font-medium pt-2">
                  Try Demo
                  <svg className="w-4 h-4 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </div>
              </div>
            </a>

            {/* Sentiment Analysis Card */}
            <a
              href="/sentiment-analysis"
              className="group relative overflow-hidden rounded-2xl border border-border bg-card p-8 transition-all hover:shadow-premium hover:scale-[1.02] hover:border-primary/50"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="relative space-y-4">
                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
                  <svg className="w-6 h-6 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h3 className="text-xl font-semibold">Sentiment Analysis</h3>
                <p className="text-muted-foreground text-sm">
                  Analyze text sentiment privately with homomorphic encryption protecting your data
                </p>
                <div className="flex items-center gap-2 text-primary text-sm font-medium pt-2">
                  Try Demo
                  <svg className="w-4 h-4 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </div>
              </div>
            </a>
          </div>
        </div>
      </section>

      <FAQSection />
      <FooterSection />
    </main>
  )
}
