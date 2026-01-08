import { HeroSection } from "@/components/sections/hero"
import { ProblemSection } from "@/components/sections/problem"
import { SolutionSection } from "@/components/sections/solution"
import { HowItWorksSection } from "@/components/sections/how-it-works"
import { FeaturesSection } from "@/components/sections/features"
import { TechnicalSpecsSection } from "@/components/sections/technical-specs"
import { UseCasesSection } from "@/components/sections/use-cases"
import { ComparisonSection } from "@/components/sections/comparison"
import { DemoSection } from "@/components/sections/demo"
import { DemoFHESection } from "@/components/sections/demo-fhe"
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
      <DemoFHESection />
      <FAQSection />
      <FooterSection />
    </main>
  )
}
