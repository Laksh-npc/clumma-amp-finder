import { Github, Mail, BookOpen } from "lucide-react";

const Footer = () => {
  return (
    <footer className="border-t bg-gradient-to-r from-muted/30 to-muted/50 mt-auto">
      <div className="container mx-auto px-4 py-8">
        <div className="grid md:grid-cols-3 gap-8 mb-6">
          {/* About */}
          <div>
            <h3 className="font-semibold text-foreground mb-3">About CLuMMA</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              A deep learning-based tool for predicting antimicrobial peptides using 
              clustering-based machine learning and attention mechanisms.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="font-semibold text-foreground mb-3">Quick Links</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="/predict" className="text-muted-foreground hover:text-primary transition-colors">
                  Prediction Tool
                </a>
              </li>
              <li>
                <a href="/research" className="text-muted-foreground hover:text-primary transition-colors">
                  Research Papers
                </a>
              </li>
              <li>
                <a href="/contact" className="text-muted-foreground hover:text-primary transition-colors">
                  Contact Us
                </a>
              </li>
              <li>
                <a href="#" className="text-muted-foreground hover:text-primary transition-colors">
                  Documentation
                </a>
              </li>
            </ul>
          </div>

          {/* Connect */}
          <div>
            <h3 className="font-semibold text-foreground mb-3">Connect</h3>
            <div className="flex gap-4">
              <a
                href="https://github.com/clumma-research"
                target="_blank"
                rel="noopener noreferrer"
                className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center hover:bg-primary/20 transition-colors"
                aria-label="GitHub"
              >
                <Github className="h-5 w-5 text-primary" />
              </a>
              <a
                href="mailto:clumma@research.edu"
                className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center hover:bg-primary/20 transition-colors"
                aria-label="Email"
              >
                <Mail className="h-5 w-5 text-primary" />
              </a>
              <a
                href="/research"
                className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center hover:bg-primary/20 transition-colors"
                aria-label="Research"
              >
                <BookOpen className="h-5 w-5 text-primary" />
              </a>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-6 border-t border-border text-center space-y-2">
          <p className="text-sm font-semibold text-foreground">
            CLuMMA - Antimicrobial Peptide Prediction Tool
          </p>
          <p className="text-xs text-muted-foreground">
            For research purposes only. Please cite our work when using this tool in publications.
          </p>
          <p className="text-xs text-muted-foreground">
            © {new Date().getFullYear()} CLuMMA Research Team. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
