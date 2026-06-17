require 'rouge' 

module Rouge 
 module Lexers 
 class MLIR < RegexLexer 
 title "MLIR" 
 desc "Multi-Level Intermediate Representation" 
 tag 'mlir' 
 filenames '*.mlir' 

 state :root do 
 rule %r/\/\/.*$/, Comment::Single 
 rule %r/\b(func|affine\.for|affine\.load|memref|tensor|linalg\.\w+)\b/, Keyword 
 rule %r/%[\w.]+/, Name::Variable 
 rule %r/[<>{}()\[\],=]/, Punctuation 
 rule %r/\d+/, Num 
 rule %r/\s+/, Text 
 rule %r/./, Text # catch-all so nothing falls into `err` 
 end 
 end 
 end 
end 