require 'rogue'

# frozen_string_literal: true

module Rouge
  module Lexers
    class MLIR < RegexLexer
      title "MLIR"
      desc "A custom language lexer for highlighting MLIR"
      tag "mlir"
      filenames "*.mlir"

      state :root do
        # Matches single line comments starting with // until the end of the line
        rule %r|//.*$|, Comment::Single
        
        # Whitespace and numbers
        rule %r/\s+/, Text
        rule %r/\d+/, Num::Integer
        
        # Strings
        rule %r/"[^"]*"/, Str
      end
    end
  end
end

