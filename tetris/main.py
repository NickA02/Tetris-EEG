# main.py

import pygame
from settings import *
from difficulties import increase_difficulty_lines_cleared
from game import Game
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Tetris (Modular)")

font = pygame.font.SysFont("Arial", 24)
clock = pygame.time.Clock()
game = Game()

increase_difficulty = increase_difficulty_lines_cleared

def draw_board(screen, board, colors):
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            color = COLORS[colors[y, x]] if colors[y, x] else COLORS["bg"]
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, COLORS["grid"], rect, 1)

def draw_piece(screen, piece, offset_x=0, offset_y=0):
    for x, y in piece.get_coords():
        if y >= 0:
            rect = pygame.Rect((x*BLOCK_SIZE)+offset_x, (y*BLOCK_SIZE)+offset_y, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, COLORS[piece.type], rect)
            pygame.draw.rect(screen, COLORS["border"], rect, 2)

def draw_mini_piece(screen, piece, pos_x, pos_y):
    # Draws a small preview of a piece at pos_x, pos_y
    shape = piece.shape
    for dy, row in enumerate(shape):
        for dx, cell in enumerate(row):
            if cell:
                rect = pygame.Rect(pos_x + dx*BLOCK_SIZE//2, pos_y + dy*BLOCK_SIZE//2, BLOCK_SIZE//2, BLOCK_SIZE//2)
                pygame.draw.rect(screen, COLORS[piece.type], rect)
                pygame.draw.rect(screen, COLORS["border"], rect, 2)

def main():
    fall_time = 0
    fall_speed = 600  # ms
    min_fall_speed = 50  # ms
    running = True
    move_delay = 80  # ms between moves when holding
    move_timer = 0
    hold_key_pressed = False
    while running:
        dt = clock.tick(60)
        fall_time += dt
        move_timer += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and not game.game_over:
                if event.key == pygame.K_UP:
                    game.rotate()
                elif event.key == pygame.K_SPACE:
                    game.hard_drop()
                elif event.key == pygame.K_c and not hold_key_pressed:
                    game.hold()
                    hold_key_pressed = True
            #Rotate does not work if held down
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_c:
                    hold_key_pressed = False

            

        keys = pygame.key.get_pressed()
        if not game.game_over:
            if move_timer > move_delay:
                if keys[pygame.K_LEFT]:
                    game.move(-1, 0)
                if keys[pygame.K_RIGHT]:
                    game.move(1, 0)
                if keys[pygame.K_DOWN]:
                    game.drop()
                move_timer = 0


        fall_speed = increase_difficulty(game)

        if fall_time > fall_speed and not game.game_over:
            game.tick()
            fall_time = 0

        screen.fill(COLORS["bg"])
        board_grid, color_grid = game.board.get_state()
        draw_board(screen, board_grid, color_grid)
        # Draw current piece
        if not game.game_over:
            draw_piece(screen, game.current_piece)
        # Draw next piece preview
        next_label = font.render("Next", True, COLORS["text"])
        screen.blit(next_label, (GRID_WIDTH*BLOCK_SIZE + 30, 30))
        draw_mini_piece(screen, game.next_piece, GRID_WIDTH*BLOCK_SIZE + 30, 60)
        # Draw hold piece preview
        hold_label = font.render("Hold (C)", True, COLORS["text"])
        screen.blit(hold_label, (GRID_WIDTH*BLOCK_SIZE + 30, 140))
        if game.hold_piece:
            draw_mini_piece(screen, game.hold_piece, GRID_WIDTH*BLOCK_SIZE + 30, 170)
        # Draw score
        score_surface = font.render(f"Score: {game.score}", True, COLORS["text"])
        screen.blit(score_surface, (GRID_WIDTH*BLOCK_SIZE + 30, 250))
        # Draw lines cleared
        lines_surface = font.render(f"Lines: {game.lines_cleared}", True, COLORS["text"])
        screen.blit(lines_surface, (GRID_WIDTH*BLOCK_SIZE + 30, 280))
        if game.game_over:
            over_surface = font.render("GAME OVER", True, (255, 0, 0))
            screen.blit(over_surface, (WINDOW_WIDTH//2 - 80, WINDOW_HEIGHT//2))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()