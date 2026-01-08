import { HeroSection } from "@/components/sections/hero"
import { ProblemSection } from "@/components/sections/problem"
import { SolutionSection } from "@/components/sections/solution"
import { HowItWorksSection } from "@/components/sections/how-it-works"
import { FeaturesSection } from "@/components/sections/features"
import { TechnicalSpecsSection } from "@/components/sections/technical-specs"
import { UseCasesSection } from "@/components/sections/use-cases"
import { ComparisonSection } from "@/components/sections/comparison"
import { DemoSection } from "@/components/sections/demo"
import { DemoFHESection } from "@/components/features/digit-recog"
import { FAQSection } from "@/components/sections/faq"
import { FooterSection } from "@/components/sections/footer"

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
      {/* Demo Call to Action */}
      <section className="py-24 bg-slate-950">
        <div className="container-custom">
          <div className="max-w-4xl mx-auto bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border border-indigo-500/20 rounded-2xl p-8 md:p-12 text-center relative overflow-hidden group">
            <div className="absolute inset-0 bg-indigo-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />

            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Ready to try it yourself?</h2>
            <p className="text-lg text-slate-400 mb-8 max-w-2xl mx-auto">
              Experience the power of Fully Homomorphic Encryption. Draw a digit and let our secure enclave classify it without ever seeing the raw data.
            </p>

            <a
              href="/digit-recog"
              className="inline-flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-8 py-3 rounded-full font-semibold transition-all hover:scale-105 shadow-lg shadow-indigo-500/25"
            >
              Launch Live Demo
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
              </svg>
            </a>
          </div>
        </div>
      </section>

      <FAQSection />
      <FooterSection />
    </main>
  )
}
