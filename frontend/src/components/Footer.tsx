const Footer = () => {
  return (
    <footer className="border-t bg-gradient-to-r from-muted/30 to-muted/50 mt-auto">
      <div className="container mx-auto px-4 py-8">
        <div className="pt-6 border-t border-border text-center space-y-2">
          <p className="text-sm font-semibold text-foreground">
            Anti Hepatitis-C Peptide Prediction Tool
          </p>
          <p className="text-xs text-muted-foreground">
            For research purposes only. Please cite our work when using this tool in publications.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
