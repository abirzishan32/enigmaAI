"use client"

import { Github, Twitter, Linkedin, Mail } from "lucide-react"

export function FooterSection() {
  return (
    <footer className="relative py-16 bg-gradient-to-b from-[#0A1628] to-[#000000] border-t border-white/10">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid md:grid-cols-4 gap-12 mb-12">
          {/* Brand */}
          <div className="md:col-span-1">
            <h3 className="text-2xl font-bold bg-gradient-to-r from-[#00D9FF] to-[#00FFB3] bg-clip-text text-transparent mb-4">
              EnigmaAI
            </h3>
            <p className="text-[#B8C5D6] text-sm leading-relaxed">
              Privacy-preserving AI framework protecting your data with cryptographic guarantees.
            </p>
          </div>

          {/* Product */}
          <div>
            <h4 className="text-white font-semibold mb-4">Product</h4>
            <ul className="space-y-2">
              {["Features", "Pricing", "Documentation", "API Reference"].map((item) => (
                <li key={item}>
                  <a href="#" className="text-[#B8C5D6] hover:text-[#00D9FF] transition-colors text-sm">
                    {item}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="text-white font-semibold mb-4">Resources</h4>
            <ul className="space-y-2">
              {["Research Paper", "GitHub", "Blog", "Community"].map((item) => (
                <li key={item}>
                  <a href="#" className="text-[#B8C5D6] hover:text-[#00D9FF] transition-colors text-sm">
                    {item}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Company */}
          <div>
            <h4 className="text-white font-semibold mb-4">Company</h4>
            <ul className="space-y-2">
              {["About", "Privacy Policy", "Terms of Service", "Contact"].map((item) => (
                <li key={item}>
                  <a href="#" className="text-[#B8C5D6] hover:text-[#00D9FF] transition-colors text-sm">
                    {item}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-white/10 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-[#B8C5D6] text-sm">
            © 2025 EnigmaAI. All rights reserved.
          </p>

          {/* Social Links */}
          <div className="flex gap-4">
            {[
              { Icon: Github, href: "#" },
              { Icon: Twitter, href: "#" },
              { Icon: Linkedin, href: "#" },
              { Icon: Mail, href: "#" }
            ].map(({ Icon, href }, index) => (
              <a
                key={index}
                href={href}
                className="w-10 h-10 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 hover:border-[#00D9FF]/50 flex items-center justify-center transition-all hover:scale-110"
              >
                <Icon className="w-5 h-5 text-[#B8C5D6] hover:text-[#00D9FF]" />
              </a>
            ))}
          </div>
        </div>
      </div>
    </footer>
  )
}
