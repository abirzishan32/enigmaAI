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


      <section className="py-24 bg-slate-950 flex justify-center">
        <GlowingStarsBackgroundCard className="max-w-4xl w-full">
          <div className="flex flex-col items-center text-center">
            <GlowingStarsTitle className="text-3xl md:text-4xl text-white mb-4">
              Ready to try it yourself?
            </GlowingStarsTitle>
            <GlowingStarsDescription className="text-lg text-slate-400 mb-8 max-w-2xl mx-auto">
              Experience the power of Fully Homomorphic Encryption. Draw a digit and let our secure enclave classify it without ever seeing the raw data.
            </GlowingStarsDescription>

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
        </GlowingStarsBackgroundCard>
      </section>

      <FAQSection />
      <FooterSection />
    </main>
  )
}
