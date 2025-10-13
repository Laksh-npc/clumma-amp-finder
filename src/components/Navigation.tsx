import { Link, useLocation } from "react-router-dom";
import { Beaker, Home, FlaskConical, BookOpen, Mail } from "lucide-react";
import { cn } from "@/lib/utils";

const Navigation = () => {
  const location = useLocation();

  const navItems = [
    { path: "/", label: "Home", icon: Home },
    { path: "/predict", label: "Prediction", icon: FlaskConical },
    { path: "/research", label: "Research", icon: BookOpen },
    { path: "/contact", label: "Contact", icon: Mail },
  ];

  return (
    <header className="sticky top-0 z-50 border-b bg-gradient-to-r from-primary via-primary/95 to-primary/90 text-primary-foreground shadow-lg backdrop-blur-sm">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 hover:opacity-90 transition-opacity">
            <Beaker className="h-8 w-8" />
            <div>
              <h1 className="text-xl font-bold tracking-tight">CLuMMA</h1>
              <p className="text-xs text-primary-foreground/80 hidden md:block">AMP Prediction Tool</p>
            </div>
          </Link>

          {/* Navigation */}
          <nav className="flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={cn(
                    "flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200",
                    "hover:bg-primary-foreground/10",
                    isActive && "bg-primary-foreground/20 font-semibold"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{item.label}</span>
                </Link>
              );
            })}
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Navigation;
